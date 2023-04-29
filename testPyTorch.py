import pandas as pd
import numpy as np
import sys
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.optim import Adam
from torch.nn import Linear, ReLU, Sequential, Sigmoid

if len(sys.argv) < 2:
    print("Usage: python3 testPyTorch.py <filename>")
    sys.exit(1)

filename = sys.argv[1]

# Load graph dataframe
graph_df = pd.read_csv('graph_files/' + filename)
#graph_df.fillna(0, inplace=True)
graph_df.fillna({'tainted': False}, inplace=True)
graph_df.fillna({'lineno:int': -1, 'endlineno:int': -1, 'childnum:int': -1}, inplace=True)
le = LabelEncoder()
graph_df['edge_type'] = le.fit_transform(graph_df['edge_type'])
graph_df['source_type'] = le.fit_transform(graph_df['source_type'])
graph_df['code'] = le.fit_transform(graph_df['code'])
#looks like this is NaN for almost every entry?
#graph_df['classname'] = le.fit_transform(graph_df['classname'])
graph_df['source_name'] = le.fit_transform(graph_df['source_name'])
#this is not a relevant feature
#graph_df['export'] = le.fit_transform(graph_df['export'])
graph_df.to_csv('processed_csv/p-'+filename, index=False)

# Define features and target
features = graph_df[['source', 'dest', 'edge_type', 'source_name', 'source_type', 'code', 'lineno:int', 'endlineno:int', 'childnum:int']].astype('float32').to_numpy()

target = graph_df['tainted'].astype('float32').to_numpy()
# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2)

# Convert the data to PyTorch tensors and create a DataLoader
train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# Define the neural network model
model = Sequential(
    Linear(features.shape[1], 16),
    ReLU(),
    Linear(16, 8),
    ReLU(),
    Linear(8, 1),
    Sigmoid()
)

# Define the loss function and optimizer
loss_fn = binary_cross_entropy_with_logits
optimizer = Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch).squeeze()
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        y_val_pred = model(torch.Tensor(X_val)).squeeze()
        val_loss = loss_fn(y_val_pred, torch.Tensor(y_val))
        print(f"Epoch {epoch+1}, val_loss: {val_loss.item():.4f}")
