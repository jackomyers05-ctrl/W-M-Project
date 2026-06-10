import pkbar
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu')

# ─── Data Loading ────────────────────────────────────────────────────────────

df = pd.read_csv("data/cmod_clean_200ms.csv", sep=',', index_col=None)
df = df.drop(columns=["Unnamed: 0"])
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df_cleaned = df.dropna(axis=1)
df_cleaned = df_cleaned.sort_values(['shot', 'time']).reset_index(drop=True)
print(df_cleaned.columns)

# ─── Normalize features (per-column z-score) ─────────────────────────────────
# Do this BEFORE building the dataset so padding zeros are meaningful (≈ mean)
feature_cols = [c for c in df_cleaned.columns if c not in ('shot', 'disruptive', 'time')]
means = df_cleaned[feature_cols].mean()
stds  = df_cleaned[feature_cols].std().replace(0, 1)   # avoid /0
df_cleaned[feature_cols] = (df_cleaned[feature_cols] - means) / stds

# ─── Dataset ─────────────────────────────────────────────────────────────────

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.sequences = []
        self.labels    = []
        self.max_length = 0

        feat_cols = [c for c in df.columns if c not in ('shot', 'disruptive', 'time')]

        for _, group in df.groupby('shot'):
            label = group['disruptive'].iloc[0]
            self.labels.append(label)

            idx_         = np.argsort(group['time'].values)
            group_sorted = group.iloc[idx_]
            features     = group_sorted[feat_cols].values.astype(np.float32)
            self.sequences.append(features)

            if features.shape[0] > self.max_length:
                self.max_length = features.shape[0]

        print("Max seq len:", self.max_length)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq   = self.sequences[idx]          # (T, F)
        label = self.labels[idx]

        # Zero-pad to max_length (safe after z-score norm — zeros ≈ mean)
        padded = np.zeros((self.max_length, seq.shape[1]), dtype=np.float32)
        padded[:len(seq)] = seq

        # True = real data, False = pad  (flipped to True=IGNORE inside train loop)
        pad_mask = np.zeros(self.max_length, dtype=np.bool_)
        pad_mask[:len(seq)] = True

        return (torch.tensor(padded,    dtype=torch.float32),
                torch.tensor(pad_mask,  dtype=torch.bool),
                torch.tensor(float(label), dtype=torch.float32))

# ─── Chronological Split + Normalize ─────────────────────────────────────────
feature_cols = [c for c in df_cleaned.columns if c not in ('shot', 'disruptive', 'time')]

all_shots = sorted(df_cleaned['shot'].unique())
n_train   = int(0.8 * len(all_shots))
train_shots = set(all_shots[:n_train])
val_shots   = set(all_shots[n_train:])

train_df = df_cleaned[df_cleaned['shot'].isin(train_shots)].reset_index(drop=True)
val_df   = df_cleaned[df_cleaned['shot'].isin(val_shots)].reset_index(drop=True)

# Normalize using train stats only
means = train_df[feature_cols].mean()
stds  = train_df[feature_cols].std().replace(0, 1)
train_df[feature_cols] = (train_df[feature_cols] - means) / stds
val_df[feature_cols]   = (val_df[feature_cols]   - means) / stds

train_dataset = CustomDataset(train_df)
val_dataset   = CustomDataset(val_df)

num_features = train_dataset.sequences[0].shape[1]
max_seq_len  = train_dataset.max_length  # pad based on train only

train_dl = DataLoader(train_dataset, batch_size=64, shuffle=True,  num_workers=0, pin_memory=True)
valid_dl = DataLoader(val_dataset,   batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

print(f"Training size:   {len(train_dataset)}")
print(f"Validation size: {len(val_dataset)}")

# ─── FF Block ────────────────────────────────────────────────────────────────

class FF(nn.Module):
    def __init__(self, embed_dim, mlp_scale: int = 2, drop_rate: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_scale),
            nn.GELU(),
            nn.Linear(embed_dim * mlp_scale, embed_dim),
            nn.Dropout(drop_rate),
        )

    def forward(self, x):
        return self.net(x)

# ─── MHSA ────────────────────────────────────────────────────────────────────

class MHSA(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, device='cuda'):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = self.head_dim ** 0.5

        self.k_proj  = nn.Linear(embed_dim, embed_dim)
        self.q_proj  = nn.Linear(embed_dim, embed_dim)
        self.v_proj  = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None, need_weights=False):
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / self.scale          # (B, H, T, T)

        if key_padding_mask is not None:
            # key_padding_mask: (B, T), True = IGNORE
            attn = attn.masked_fill(key_padding_mask[:, None, None, :], -1e9)
            # Use -1e9 instead of -inf to avoid NaN when entire rows are masked

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return out, (attn if need_weights else None)

# ─── EncoderBlock ─────────────────────────────────────────────────────────────

class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_scale: int = 2,
                 drop_rate: float = 0.0, device='cuda'):
        super().__init__()
        self.LN1    = nn.LayerNorm(embed_dim)
        self.attn   = MHSA(embed_dim, num_heads, dropout=drop_rate, device=device)
        self.c_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.LN2    = nn.LayerNorm(embed_dim)
        self.ff     = FF(embed_dim, mlp_scale=mlp_scale, drop_rate=drop_rate)

    def forward(self, x, padding_mask=None, need_weights=False):
        attn_out, attn_weights = self.attn(
            self.LN1(x), key_padding_mask=padding_mask, need_weights=need_weights
        )
        x = x + self.c_proj(attn_out)
        x = x + self.ff(self.LN2(x))
        return x

# ─── Classifier ──────────────────────────────────────────────────────────────

class Classifier(nn.Module):
    def __init__(self, num_features, embed_dim, num_classes=1,
                 attn_heads=[4, 8, 8], mlp_scale: int = 2,
                 drop_rates=[0.0, 0.0, 0.0], device='cuda', max_seq_len=182):
        super().__init__()
        assert len(drop_rates) == len(attn_heads)

        self.embedding     = nn.Linear(num_features, embed_dim)
        self.cls_token     = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len + 1, embed_dim))

        self.layers = nn.ModuleList([
            EncoderBlock(embed_dim, attn_heads[i], mlp_scale,
                         drop_rate=drop_rates[i], device=device)
            for i in range(len(attn_heads))
        ])

        self.mlp_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x, padding_mask=None):
        B = x.size(0)

        x = self.embedding(x)                                       # (B, T, D)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)  # (B, T+1, D)
        x = x + self.pos_embedding[:, :x.size(1)]

        if padding_mask is not None:
            cls_mask     = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            padding_mask = torch.cat([cls_mask, padding_mask], dim=1)

        for layer in self.layers:
            x = layer(x, padding_mask=padding_mask)

        return self.mlp_head(x[:, 0]).squeeze(-1)                   # (B,)

# ─── Model Init ──────────────────────────────────────────────────────────────

embed_dim   = 256
mlp_scale   = 2
drop_rates  = [0.1, 0.1, 0.1]
attn_heads  = [4, 8, 8]
num_classes = 1

model = Classifier(
    num_features=num_features,
    embed_dim=embed_dim,
    num_classes=num_classes,
    attn_heads=attn_heads,
    mlp_scale=mlp_scale,
    drop_rates=drop_rates,
    device=device,
    max_seq_len=max_seq_len,
)
model.to(device)
print(f"Network Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ─── Training ────────────────────────────────────────────────────────────────

def train(model, num_epochs, train_dl, valid_dl, lr=1e-4, device='cuda', seed=8):
    model.to(device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    # Class-weighted loss for imbalanced disruption data
    all_labels  = torch.tensor([y for _, _, y in train_dl.dataset])
    pos         = all_labels.sum().item()
    neg         = len(all_labels) - pos
    pos_weight  = torch.tensor([neg / pos], device=device)
    print(f"Class balance — pos: {int(pos)}, neg: {int(neg)}, pos_weight: {pos_weight.item():.2f}")

    loss_fn   = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_hist_train,     loss_hist_valid     = [0] * num_epochs, [0] * num_epochs
    accuracy_hist_train, accuracy_hist_valid = [0] * num_epochs, [0] * num_epochs
    f1_hist_train,       f1_hist_valid       = [0] * num_epochs, [0] * num_epochs

    for epoch in range(num_epochs):
        model.train()
        kbar = pkbar.Kbar(target=len(train_dl), epoch=epoch, num_epochs=num_epochs,
                          width=20, always_stateful=False)

        total_loss    = 0
        all_preds_tr  = []
        all_labels_tr = []

        for i, (x_batch, pad_mask, y_batch) in enumerate(train_dl):
            x_batch  = x_batch.to(device)
            y_batch  = y_batch.to(device)
            # Flip: dataset True=real → model expects True=IGNORE
            padding_mask = (~pad_mask).to(device)

            pred = model(x_batch, padding_mask=padding_mask)
            loss = loss_fn(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            preds_bin  = (pred > 0).float()
            batch_acc  = (preds_bin == y_batch).float().mean().item()
            total_loss += loss.item() * y_batch.size(0)

            all_preds_tr.extend(preds_bin.cpu().tolist())
            all_labels_tr.extend(y_batch.cpu().tolist())

            kbar.update(i, values=[("loss", loss.item()), ("acc", batch_acc)])

        loss_hist_train[epoch]     = total_loss / len(train_dl.dataset)
        accuracy_hist_train[epoch] = (np.array(all_preds_tr) == np.array(all_labels_tr)).mean()
        f1_hist_train[epoch]       = f1_score(all_labels_tr, all_preds_tr, zero_division=0)

        # ── Validation ───────────────────────────────────────────────────────
        model.eval()
        val_loss       = 0
        all_preds_val  = []
        all_labels_val = []

        with torch.no_grad():
            for x_batch, pad_mask, y_batch in valid_dl:
                x_batch      = x_batch.to(device)
                y_batch      = y_batch.to(device)
                padding_mask = (~pad_mask).to(device)

                pred     = model(x_batch, padding_mask=padding_mask)
                val_loss += loss_fn(pred, y_batch).item() * y_batch.size(0)

                preds_bin = (pred > 0).float()
                all_preds_val.extend(preds_bin.cpu().tolist())
                all_labels_val.extend(y_batch.cpu().tolist())

        loss_hist_valid[epoch]     = val_loss / len(valid_dl.dataset)
        accuracy_hist_valid[epoch] = (np.array(all_preds_val) == np.array(all_labels_val)).mean()
        f1_hist_valid[epoch]       = f1_score(all_labels_val, all_preds_val, zero_division=0)

        print(
            f"Epoch {epoch+1:02d} | "
            f"Train  Loss: {loss_hist_train[epoch]:.4f}  Acc: {accuracy_hist_train[epoch]:.4f}  F1: {f1_hist_train[epoch]:.4f} | "
            f"Val    Loss: {loss_hist_valid[epoch]:.4f}  Acc: {accuracy_hist_valid[epoch]:.4f}  F1: {f1_hist_valid[epoch]:.4f}"
        )

    return loss_hist_train, loss_hist_valid, accuracy_hist_train, accuracy_hist_valid, f1_hist_train, f1_hist_valid

# ─── Run ─────────────────────────────────────────────────────────────────────

num_epochs = 10
hist = train(model, num_epochs, train_dl, valid_dl)

# ─── Evaluation: Graphs, Metrics, Prediction Testing ────────────────────────
# Run this cell AFTER training completes and hist is returned

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve,
    auc, precision_recall_curve, average_precision_score
)
import warnings
warnings.filterwarnings('ignore')

# Unpack history
loss_tr, loss_val, acc_tr, acc_val, f1_tr, f1_val = hist
epochs = range(1, len(loss_tr) + 1)

# ─── 1. Training Curves ───────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Training History — CMod Disruption Classifier", fontsize=13, fontweight='bold')

axes[0].plot(epochs, loss_tr,  'o-', label='Train', color='#2563eb')
axes[0].plot(epochs, loss_val, 'o-', label='Val',   color='#f97316', linestyle='--')
axes[0].set_title('Loss');  axes[0].set_xlabel('Epoch'); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(epochs, acc_tr,  'o-', label='Train', color='#2563eb')
axes[1].plot(epochs, acc_val, 'o-', label='Val',   color='#f97316', linestyle='--')
axes[1].set_title('Accuracy'); axes[1].set_xlabel('Epoch'); axes[1].legend(); axes[1].grid(alpha=0.3)

axes[2].plot(epochs, f1_tr,  'o-', label='Train', color='#2563eb')
axes[2].plot(epochs, f1_val, 'o-', label='Val',   color='#f97316', linestyle='--')
axes[2].set_title('F1 Score'); axes[2].set_xlabel('Epoch'); axes[2].legend(); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
plt.show()

# ─── 2. Collect Val Predictions (raw logits + labels) ────────────────────────

model.eval()
all_logits = []
all_labels = []
all_probs  = []

with torch.no_grad():
    for x_batch, pad_mask, y_batch in valid_dl:
        x_batch      = x_batch.to(device)
        padding_mask = (~pad_mask).to(device)

        logits = model(x_batch, padding_mask=padding_mask)
        probs  = torch.sigmoid(logits)

        all_logits.extend(logits.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
        all_labels.extend(y_batch.tolist())

all_logits = np.array(all_logits)
all_probs  = np.array(all_probs)
all_labels = np.array(all_labels)
all_preds  = (all_logits > 0).astype(int)

# ─── 3. Classification Report ─────────────────────────────────────────────────

print("=" * 55)
print("       CLASSIFICATION REPORT — Validation Set")
print("=" * 55)
print(classification_report(all_labels, all_preds,
      target_names=['Stable', 'Disruptive'], digits=4))

# ─── 4. Confusion Matrix ──────────────────────────────────────────────────────

cm = confusion_matrix(all_labels, all_preds)
tn, fp, fn, tp = cm.ravel()

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(['Stable', 'Disruptive']); ax.set_yticklabels(['Stable', 'Disruptive'])
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix — Validation Set', fontweight='bold')
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center',
                color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=14)
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nTP: {tp}  FP: {fp}  TN: {tn}  FN: {fn}")
print(f"Missed disruptions (FN): {fn}  |  False alarms (FP): {fp}")

# ─── 5. ROC + PR Curves ───────────────────────────────────────────────────────

fpr, tpr, roc_thresh = roc_curve(all_labels, all_probs)
roc_auc              = auc(fpr, tpr)

prec, rec, pr_thresh = precision_recall_curve(all_labels, all_probs)
ap                   = average_precision_score(all_labels, all_probs)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

ax1.plot(fpr, tpr, color='#2563eb', lw=2, label=f'AUC = {roc_auc:.4f}')
ax1.plot([0, 1], [0, 1], 'k--', lw=1)
ax1.set_xlabel('False Positive Rate'); ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curve', fontweight='bold'); ax1.legend(); ax1.grid(alpha=0.3)

ax2.plot(rec, prec, color='#f97316', lw=2, label=f'AP = {ap:.4f}')
ax2.axhline(y=all_labels.mean(), color='k', linestyle='--', lw=1, label='Baseline (prevalence)')
ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision')
ax2.set_title('Precision-Recall Curve', fontweight='bold'); ax2.legend(); ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('roc_pr_curves.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nROC-AUC: {roc_auc:.4f}  |  Average Precision: {ap:.4f}")

# ─── 6. Threshold Sweep — find best F1 threshold ─────────────────────────────

from sklearn.metrics import f1_score as f1s

thresholds  = np.linspace(0.01, 0.99, 200)
f1_scores   = [f1s(all_labels, (all_probs > t).astype(int), zero_division=0) for t in thresholds]
best_thresh = thresholds[np.argmax(f1_scores)]
best_f1     = max(f1_scores)

plt.figure(figsize=(7, 3))
plt.plot(thresholds, f1_scores, color='#2563eb', lw=2)
plt.axvline(best_thresh, color='#f97316', linestyle='--', label=f'Best threshold = {best_thresh:.2f}')
plt.xlabel('Decision Threshold'); plt.ylabel('F1 Score')
plt.title('F1 vs Decision Threshold', fontweight='bold')
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('threshold_sweep.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nDefault threshold (0.5) F1:  {f1s(all_labels, (all_probs > 0.5).astype(int)):.4f}")
print(f"Best threshold ({best_thresh:.2f})  F1:  {best_f1:.4f}")

# ─── 7. Per-Shot Prediction Testing ──────────────────────────────────────────
# Grab a few individual shots from the validation set and show predictions

print("\n" + "=" * 55)
print("         PER-SHOT PREDICTION TESTING")
print("=" * 55)

model.eval()
results = []

with torch.no_grad():
    for idx in range(len(val_dataset)):
        seq, mask, label = val_dataset[idx]
        x    = seq.unsqueeze(0).to(device)
        pm   = (~mask.unsqueeze(0)).to(device)
        logit = model(x, padding_mask=pm).item()
        prob  = torch.sigmoid(torch.tensor(logit)).item()
        pred  = int(logit > 0)
        results.append({
            'idx':    idx,
            'label':  int(label.item()),
            'pred':   pred,
            'prob':   prob,
            'correct': pred == int(label.item()),
        })

# Show sample of each category
import random
random.seed(42)

def show_samples(title, items, n=5):
    sample = random.sample(items, min(n, len(items)))
    print(f"\n── {title} (showing {len(sample)}) ──")
    print(f"  {'Shot idx':>8}  {'True':>6}  {'Pred':>6}  {'P(disrupt)':>11}  {'✓/✗':>4}")
    for r in sample:
        mark = '✓' if r['correct'] else '✗'
        true_lbl = 'DISRUPT' if r['label'] else 'stable'
        pred_lbl = 'DISRUPT' if r['pred']  else 'stable'
        print(f"  {r['idx']:>8}  {true_lbl:>7}  {pred_lbl:>7}  {r['prob']:>10.3f}  {mark:>4}")

tp_list = [r for r in results if r['label']==1 and r['pred']==1]
tn_list = [r for r in results if r['label']==0 and r['pred']==0]
fp_list = [r for r in results if r['label']==0 and r['pred']==1]
fn_list = [r for r in results if r['label']==1 and r['pred']==0]

show_samples("True Positives  (caught disruptions)",   tp_list)
show_samples("True Negatives  (correctly stable)",     tn_list)
show_samples("False Positives (false alarms)",         fp_list)
show_samples("False Negatives (missed disruptions)",   fn_list)

print(f"\nSummary: {len(tp_list)} TP | {len(fp_list)} FP | {len(tn_list)} TN | {len(fn_list)} FN")
print(f"Missed disruptions: {len(fn_list)} / {len(tp_list)+len(fn_list)} total disruptive shots")
