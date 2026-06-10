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

### Random Walk testing

# Specific imports
import random as r
import pandas as pd
# File prototyping "random walk" mechanic until I have the full training script
# Idea is to train the transformer on a fixed random seed, then shuffle the dataset
#   using different random seeds, then testing it. 
# Need to generate array of random intervals between 1 and 100

r_seed_list = r.sample(range(1,1001),100)

for seed in r_seed_list:
    # Set seed for a run
    torch.manual_seed(seed)
    
    # Re-load dataset
    df = pd.read_csv("data/cmod_clean_200ms.csv", sep=',', index_col=None) # fix for needs
    
    # Shuffle dataset
    df = df.sample(frac=1.0)
    
    # Placeholder dataloading 
    features = df.iloc[:, :-1].values  # All columns except last, returns as a numpy array. 
    target = df.iloc[:, -1].values     # Last column is disruption label (0 or 1)
    scaler = StandardScaler()
    features = scaler.fit_transform(features) # failing due to inf values
    total_size = int(len(dataset)) # define dataset
    test_size = int(0.15 * total_size)
    test_dl = features[0: (test_size)]
    
    # Set model to evaluation mode
    model.eval()
    
    # Define specific variables
    val_loss       = 0
    all_preds_val  = []
    all_labels_val = []
    
    # Get batch for inputs
    x_batch, pad_mask, y_batch = enumerate(test_dl)

    # Run section from production model
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

    # Print current validation loss
    print(
        f"Epoch {epoch+1:02d} | "
        f"Val    Loss: {loss_hist_valid[epoch]:.4f}  Acc: {accuracy_hist_valid[epoch]:.4f}  F1: {f1_hist_valid[epoch]:.4f}"
        )
    
    