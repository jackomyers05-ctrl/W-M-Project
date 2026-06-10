### Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility 
torch.manual_seed(1) 

# Load data
df = pd.read_csv("data/cmod_clean_200ms.csv", sep=',', index_col=None)

# Remove all data with NaN or inf values. 
# pseudocode, idk bro

# View data size and observe data visually, ground-truth
print(df.shape)
df.hist(figsize=(15,15),bins=50,density=True)

# Extract features (64 raw values) and target (disruption label)
features = df.iloc[:, :-1].values  # All columns except last, returns as a numpy array. 
target = df.iloc[:, -1].values     # Last column is disruption label (0 or 1)
print(features.shape) 
print(target.shape)

#features.hist(figsize=(15,15),bins=50,density=True) # check data in the dataset

# Normalize features
scaler = StandardScaler()
features = scaler.fit_transform(features) # failing due to inf values

# 70/15/15 split, features
total_size = int(len(df))
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = int(0.15 * total_size)
# Selection idx order is train->val->test
train_dataset_feature = features[0:train_size]
val_dataset_feature = features[train_size: (train_size+val_size)]
test_dataset_feature = features[(train_size + val_size): (total_size)]

# 70/15/15 split, target
total_size = int(len(df))
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = int(0.15 * total_size)
# Selection idx order is train->val->test
train_dataset_target = target[0:train_size]
val_dataset_target = target[train_size: (train_size+val_size)]
test_dataset_target = target[(train_size + val_size): (total_size)]

# Dataset class
class TransformerDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x = torch.tensor(x_data, dtype=torch.float32)
        self.y = torch.tensor(y_data, dtype=torch.long)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)

# Create dataloader
dataset = TransformerDataset(train_dataset_feature, train_dataset_target)
# can also be "dataset = LSTMDataset(features, target)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model components

class FF(nn.Module):
    def __init__(self,embed_dim, mlp_scale : int = 2, drop_rate: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim, embed_dim*mlp_scale)
        self.g1 = nn.GELU()
        self.ln2 = nn.LayerNorm(embed_dim*mlp_scale, embed_dim)
        self.drop = nn.Dropout(drop_rate)

    def forward(self,x):
        x = self.ln1(x)
        x = self.g1(x)
        x = self.ln2(x)
        x = self.drop(x)
        return x

class MHSA(nn.Module):
    def __init__(self, embed_dim, num_heads, seq_len=250, dropout=0.2,device='cuda'):
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim is indivisible by num_heads"

        self.num_heads = num_heads
        self.seq_length = seq_len
        self.head_dim = embed_dim // num_heads
        self.d_k = self.head_dim ** 0.5
        self.device = device
        # Create the projections for Q,K,V
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x,attn_mask=None,key_padding_mask=None,need_weights=False):
        batch_size, seq_len, embed_dim = x.shape

        # Project input x to queries, keys, and values
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Shaping Q, K, and V tensors
        k = tensor.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = tensor.view(batch_size, seq_len, self.num_heads, self.head_dim)
        q = tensor.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transposing for alignment, don't change
        k = torch.transpose(k, 1, 2)
        q = torch.transpose(q, 1, 2)
        v = torch.transpose(v, 1, 2)

        # Attention determination, don't change
        K = torch.transpose(k, 2, 3)
        Q = q
        attn_scores = Q @ K^T / self.d_k

        # Option for masked
        if attn_mask is not None:
            attn_scores.masked_fill_(attn_mask,-torch.inf)

        # Option for adding padding
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask[:, None, None, :]
            attn_scores.masked_fill_(key_padding_mask,-torch.inf)

        # Compute the softmax over attention
        attn_scores = F.softmax(attn_scores, dim=-1)

        # Apply dropout
        attn_scores = self.dropout(attn_scores)

        # Apply V weighting
        attn_output = attn_scores @ v

        # Transpose for alignment, don't change
        attn_output = torch.transpose(attn_output, 1, 2)

        # Flatten back out to 1 dimension
        attn_output = attn_output.contiguous().view(batch_size,seq_len,embed_dim)

        if need_weights:
            return attn_output,attn_scores
        else:
            return attn_output,None

class EncoderBlock(nn.Module):
    def __init__(self,embed_dim,num_heads, mlp_scale : int = 2,drop_rate: float = 0.2, device='cuda'):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.device = device
        self.mlp_scale = mlp_scale
        self.drop_rate = drop_rate
        self.LN1 = nn.LayerNorm(embed_dim)
        self.attn = MHSA(embed_dim=embed_dim, num_heads=num_heads, seq_len=250, dropout=drop_rate, device=device)
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.LN2 = nn.LayerNorm(embed_dim)
        # Feed forward
        self.FF = FF(embed_dim=embed_dim, mlp_scale=mlp_scale, drop_rate=drop_rate)

    # Mask generation, unused
    def generate_mask(self,seq_len):
        return torch.triu(torch.ones((seq_len, seq_len), device=self.device, dtype=torch.bool), diagonal=1)

    def forward(self, x,padding_mask=None,need_weights=False):
        B,N_t,t_dim = x.shape
        x_norm = self.Ln1(x)
        
        # Calculate attention and weights
        attn,attn_weights = self.attn(x_norm, key_padding_mask=None, need_weights=False)

        attn = self.c_proj(attn) # Optional projection layer - done for you

        # Add attention
        x = x + attn
        # Calculate residual to add
        res = self.FF(self.LN2(x))
        # Add residual
        x = x + res
        return x

### Vision model class, MUST be converted back to regular transformer using nn.Transformer

class ViT(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim, num_classes=10,
                 attn_heads=[2, 4, 2], mlp_scale: int = 2, drop_rates=[0.0, 0.0, 0.0], device='cuda'):
        super().__init__()

        # Ensure drop_rates and attn_heads lists are consistent
        assert len(drop_rates) == len(attn_heads), "drop_rates and attn_heads must be of same length"
        # Ensure the image dimensions are divisible by the patch size
        assert image_size[1] % patch_size == 0 and image_size[2] % patch_size == 0, \
            'image dimensions must be divisible by the patch size'

        channels = image_size[0]
        # Compute the number of patches by dividing the height and width by patch size
        num_patches = (image_size[1] // patch_size) * (image_size[2] // patch_size)
        # Each patch is flattened into a vector of size (channels * patch_size^2)
        patch_dim = channels * patch_size ** 2
        self.device = device

        # Patch embedding layer:
        # 1. Rearranges the image into a sequence of flattened patches
        # 2. Applies LayerNorm to each patch
        # 3. Projects to the embedding dimension using a Linear layer
        # 4. Applies another LayerNorm
        self.patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # Positional embedding to retain spatial information (learned parameter)
        # Shape: (1, num_patches + 1, embed_dim); +1 for the [CLS] token
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

        # Create a list of transformer encoder blocks, each with a different number of attention heads
        # and optional dropout
        layers_ = [
            EncoderBlock(embed_dim, attn_heads[i], mlp_scale, drop_rate=drop_rates[i])
            for i in range(len(attn_heads))
        ]
        self.layers = nn.ModuleList(layers_)

        # Classification head to project the [CLS] token's embedding to logits for each class
        self.mlp_head = nn.Linear(embed_dim, num_classes)

        # Learnable [CLS] token used for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x, padding_mask=None):
        batch_size = x.size(0)

        # Convert image into a sequence of embedded patches
        x = self.patch_embedding(x)  # Shape: (B, num_patches, embed_dim)

        # Expand the [CLS] token to match the batch size
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # Shape: (B, 1, embed_dim)

        # Prepend the [CLS] token to the patch embeddings
        x = torch.cat((cls_tokens, x), dim=1)  # Shape: (B, num_patches + 1, embed_dim)

        # Add positional embeddings
        x = x + self.pos_embedding[:, :x.size(1)]  # Shape: (B, num_patches + 1, embed_dim)

        # Pass through the stack of transformer encoder blocks
        for layer in self.layers:
            x = layer(x)

        # Use the [CLS] token's output for classification
        return self.mlp_head(x[:, 0])  # Shape: (B, num_classes)


### Training script

image_size = train_dataset.__getitem__(0)[0].shape
patch_size = 4 # Patch size - grab patch x patch pixels to form an element in the sequence
embed_dim = 64 # embedding dimension - each patch is projected in a space of dim = embed_dim
mlp_scale = 2 # instead of specifying the MLP dimensions explicitly, scale it based off of embed_dim
drop_rates = [0.0,0.0,0.0] # applied for each encoderblock
attn_heads = [2,2,2] # 3 encoder blocks, each with 2 heads -> embed_dim // attn_heads[i] should = 0
num_classes = len(mnist_classes)
model = ViT(image_size,patch_size,embed_dim,num_classes=num_classes,attn_heads=attn_heads,mlp_scale=mlp_scale,drop_rates=drop_rates,device=device)
t_params = sum(p.numel() for p in model.parameters())
print("Network Parameters: ",t_params)
model.to(device)

def train(model, num_epochs, train_dl, valid_dl,lr=0.001,device='cuda',seed=8):
    model.to(device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_hist_train = [0] * num_epochs
    accuracy_hist_train = [0] * num_epochs
    loss_hist_valid = [0] * num_epochs
    accuracy_hist_valid = [0] * num_epochs

    for epoch in range(num_epochs):
        model.train()
        kbar = pkbar.Kbar(target=len(train_dl), epoch=epoch, num_epochs=num_epochs, width=20, always_stateful=False)

        total_correct = 0
        total_loss = 0

        for i, (x_batch, y_batch) in enumerate(train_dl):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            batch_correct = (torch.argmax(pred, dim=1) == y_batch).float().sum()
            total_correct += batch_correct.item()
            total_loss += loss.item() * y_batch.size(0)

            kbar.update(i, values=[("loss", loss.item()), ("acc", batch_correct.item() / y_batch.size(0))])

        loss_hist_train[epoch] = total_loss / len(train_dl.dataset)
        accuracy_hist_train[epoch] = total_correct / len(train_dl.dataset)

        model.eval()
        val_loss = 0
        val_correct = 0
        with torch.no_grad():
            for x_batch, y_batch in valid_dl:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)
                val_loss += loss.item() * y_batch.size(0)
                val_correct += (torch.argmax(pred, dim=1) == y_batch).float().sum().item()

        loss_hist_valid[epoch] = val_loss / len(valid_dl.dataset)
        accuracy_hist_valid[epoch] = val_correct / len(valid_dl.dataset)

        kbar.add(1, values=[("val_loss", loss_hist_valid[epoch]), ("val_acc", accuracy_hist_valid[epoch])])

    return loss_hist_train, loss_hist_valid, accuracy_hist_train, accuracy_hist_valid

### Training function call

num_epochs = 10
hist = train(model, num_epochs, train_dl, valid_dl)

### Training loss

def plot_loss(hist):
    x_arr = np.arange(len(hist[0])) + 1  # number of epochs

    fig = plt.figure(figsize=(12, 4))

    ax = fig.add_subplot(1, 2, 1)
    ax.plot(x_arr, hist[0], '-o', label='Train loss')
    ax.plot(x_arr, hist[1], '--<', label='Validation loss')
    ax.set_xlabel('Epoch', size=15)
    ax.set_ylabel('Loss', size=15)
    ax.legend(fontsize=15)

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(x_arr, hist[2], '-o', label='Train acc.')
    ax.plot(x_arr, hist[3], '--<', label='Validation acc.')
    ax.legend(fontsize=15)
    ax.set_xlabel('Epoch', size=15)
    ax.set_ylabel('Accuracy', size=15)
    plt.show()

plot_loss(hist)


