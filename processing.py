import pandas as pd
import numpy as np
import sys
import os
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.optim import Adam
from torch.nn import Linear, ReLU, Sequential, Sigmoid

# set up batch size and number of epochs
batch_size = 64
num_epochs = 10

# set up folder path containing all the csv files
folder_path = "graph_files/"

# iterate over all csv files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        # read csv file into pandas dataframe
        df = pd.read_csv(os.path.join(folder_path, filename))

        # preprocess dataframe and convert to PyTorch tensors
        df.fillna({'tainted': False}, inplace=True)
        df.fillna({'lineno:int': -1, 'endlineno:int': -1, 'childnum:int': -1}, inplace=True)
        le = LabelEncoder()
        df['edge_type'] = le.fit_transform(df['edge_type'])
        df['source_type'] = le.fit_transform(df['source_type'])
        df['code'] = le.fit_transform(df['code'])
        #looks like this is NaN for almost every entry?
        #graph_df['classname'] = le.fit_transform(df['classname'])
        df['source_name'] = le.fit_transform(df['source_name'])
        #this is not a relevant feature
        #graph_df['export'] = le.fit_transform(graph_df['export'])
        df.to_csv('processed_csv/p-'+filename, index=False)

        # Define features and target
        features = df[['source', 'dest', 'edge_type', 'source_name', 'source_type', 'lineno:int', 'endlineno:int', 'childnum:int', 'code']].astype('float32').to_numpy()
        
        target = df['tainted'].astype('float32').to_numpy()
        
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

        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2)

        # Convert the data to PyTorch tensors and create a DataLoader
        train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

        val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        # train model on current dataset
        for epoch in range(num_epochs):
            for i, (X_batch, y_batch) in enumerate(train_loader):
                model.train()
                optimizer.zero_grad()

                # forward pass
                y_pred = model(X_batch).squeeze()

                # calculate loss
                loss = loss_fn(y_pred, y_batch)

                # backward pass and optimization step
                loss.backward()
                optimizer.step()

            # evaluate model on validation set
            with torch.no_grad():
                model.eval()
                val_loss = 0
                for data in val_loader:
                    y_pred_val = model(X_val)
                    val_loss += loss_fn(y_pred_val, y_val).item()
                val_loss /= len(val_loader)

            # print progress
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")

        # save trained model for current dataset
        torch.save(model.state_dict(), f"{filename[:-4]}_model.pth")
