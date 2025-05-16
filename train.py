import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm
import hydra

from sklearn.model_selection import train_test_split

from .model import MLPModel
from .dataset import Dataset
from .utils import print_metrics

class Trainer:
    def __init__(self, criteriion:str, device:str):
        self.model = MLPModel(device=device)
        self.optimizer = optim.Adam(self.model.parameters())

        if criteriion == "BCEWithLogits":
            self.criteriion = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError()

    def train(self, train_loader:DataLoader, val_loader:DataLoader, load_path:str=None, save_path:str=None, num_epochs:int=None, patience:int=10000):
        self.model.load(path=load_path)
        history = {'train_loss': [], 'val_loss': [], 'val_auc': [], 'val_accuracy': []}
        batch_iterator = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in tqdm(range(num_epochs)):
            self.model.train()
            train_loss = 0.0

            for features, labels in train_loader:
                labels_pre = self.model(features)
                loss = self.criteriion(labels_pre, labels)
                self.optimizer.zero_grad()
                loss.zero_grad()
                self.optimizer.step()
                train_loss += loss.item()
                batch_iterator.set_postfix(loss=loss.item())
            
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            tqdm.write(f"Epoch {epoch+1}: train_loss = {train_loss:.4f}")

            self.model.eval()
            val_loss = 0.0
            all_labels = []
            all_outputs_prob = []

            with torch.no_grad():
                for features, labels in val_loader:
                    labels_pre = self.model(features)
                    loss = self.criteriion(labels_pre, labels)
                    val_loss += loss

                    all_labels.extend(labels.cpu().numpy())
                    all_outputs_prob.extend(torch.sigmoid(labels_pre).cpu().numpy())
                val_loss /= len(val_loader)
            
            history['val_loss'].append(val_loss)
            all_labels = np.array(all_labels).flatten()
            all_outputs_proba = np.array(all_outputs_proba).flatten()
            all_outputs_binary = (all_outputs_proba > 0.5).astype(int)

            metrics = print_metrics(all_labels, all_outputs_proba, all_outputs_binary, phase=f"Epoch {epoch+1}/{num_epochs} Val")
            history['val_auc'].append(metrics['auc'])
            history['val_accuracy'].append(metrics['accuracy'])

            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                self.model.save(path=save_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve > patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break
        
        print("Training complete.")
        return history

    def predict(model, data_loader:DataLoader):
        model.eval()
        predictions_prob = []
        with torch.no_grad():
            for features in data_loader:
                if isinstance(features, list):
                    features = features[0]
                outputs = model(features)
                predictions_prob.extend(torch.sigmoid(outputs).cpu().numpy())
        return np.array(predictions_prob).flatten()
    
@hydra.main(config_name="train", config_path="config", version_base="1.3")
def main(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # STEP1: Training
    train_val_df = pd.read_csv(os.path.join(config.data.path, config.data.train_file), index_col='customer_id')

    X = train_val_df.drop("???", axis=1)
    y = train_val_df["???"]

    X = X.replace([np.inf, -np.inf], np.nan)
    for col in X.columns:
        if X[col].isnull().any():
            print(f"Warning: Column '{col}' contains NaN values. Imputing with mean.")
            X[col] = X[col].fillna(X[col].mean())

    X_train_df, X_val_df, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=config.train.seed, stratify=y)

    train_dataset = Dataset(X_train_df, y_train, scaler=scaler, fit_scaler=True)
    fitted_scaler = train_dataset.get_scaler()
    val_dataset = Dataset(X_val_df, y_val, scaler=fitted_scaler, fit_scaler=False)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    
    trainer = Trainer(device=device)
    train_history = trainer.train(train_loader=train_loader, val_loader=val_loader, load_path=config.model.load_path, save_path=config.model.save_pth, num_epochs=config.train.num_epochs, patience=config.train.patience)

    #STEP2: Testing
    test_df_pytorch = pd.read_csv(os.path.join(config.data.path, config.data.test_file), index_col='customer_id')
    test_df_pytorch = test_df_pytorch.replace([np.inf, -np.inf], np.nan)
    for col in test_df_pytorch.columns:
        if test_df_pytorch[col].isnull().any():
                test_df_pytorch[col] = test_df_pytorch[col].fillna(X[col].mean())

    test_dataset_pytorch = Dataset(test_df_pytorch, labels=None, scaler=fitted_scaler, fit_scaler=False)
    test_loader_pytorch = DataLoader(test_dataset_pytorch, batch_size=256, shuffle=False)
    
    test_pre = trainer.predict(test_loader_pytorch)

    # STEP3: Create a submission file
    submission_pytorch_df = pd.DataFrame({'customer_id': test_df_pytorch.index, 'prediction_pytorch': test_pre})
    submission_pytorch_df.to_csv('submission_pytorch.csv', index=False)
    print("PyTorch submission_pytorch.csv saved.")

if __name__ == "__main__":
    main()