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
total_size = int(len(dataset))
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = int(0.15 * total_size)
# Selection idx order is train->val->test
train_dataset = features[0:train_size]
val_dataset = features[train_size: (train_size+val_size)]
test_dataset = features[(train_size + val_size): (total_size)]

# 70/15/15 split, target
total_size = int(len(dataset))
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = int(0.15 * total_size)
# Selection idx order is train->val->test
train_dataset = target[0:train_size]
val_dataset = target[train_size: (train_size+val_size)]
test_dataset = target[(train_size + val_size): (total_size)]


# Create sequences (sliding windows for LSTM), Joseph: suggest deletion since we're not doing time-sequential data
def create_sequences(data, labels, seq_len=50):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(labels[i+seq_len])
    return np.array(X), np.array(y)

# would be good for when we do time-sequence
x_np, y_np = create_sequences(features, target, seq_len=50)

# Dataset class
class LSTMDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x = torch.tensor(x_data, dtype=torch.float32)
        self.y = torch.tensor(y_data, dtype=torch.long)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)

# Create dataloader
dataset = LSTMDataset(x_np, y_np)
# can also be "dataset = LSTMDataset(features, target)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# MODEL
