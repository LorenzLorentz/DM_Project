import os
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm
import hydra

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from model import MLPModel
from dataset import Dataset
from utils import print_metrics

class Trainer:
    def __init__(self, criterion:str="BCEWithLogits", device:str=None):
        self.device = device
        self.model = MLPModel(device=device)
        self.optimizer = optim.Adam(self.model.parameters())

        if criterion == "BCEWithLogits":
            self.criterion = nn.BCEWithLogitsLoss(torch.Tensor([100.0], device=device))
        else:
            raise NotImplementedError()

    def train(self, train_loader:DataLoader, val_loader:DataLoader, load_path:str=None, save_path:str=None, num_epochs:int=None, patience:int=10000):
        self.model.load(path=load_path)
        history = {'train_loss': [], 'val_loss': [], 'val_auc': [], 'val_accuracy': [], 'val_precision': []}
        best_val_auc = -float('inf')
        epochs_no_improve = 0

        epoch_iterator = tqdm(range(num_epochs), desc="Epochs")
        for epoch in epoch_iterator:
            batch_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Train", leave=False)
            self.model.train()
            train_loss = 0.0

            for features, labels in batch_iterator:
                labels_pre = self.model(features)
                loss = self.criterion(labels_pre, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                batch_iterator.set_postfix(loss=loss.item())
            
            train_loss /= len(train_loader.dataset)
            history['train_loss'].append(train_loss)

            self.model.eval()
            val_loss = 0.0
            all_labels = []
            all_outputs_prob = []

            val_batch_iterator = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Validation", leave=False)
            with torch.no_grad():
                for features, labels in val_batch_iterator:
                    labels_pre = self.model(features)
                    loss = self.criterion(labels_pre, labels)
                    val_loss += loss

                    all_labels.extend(labels.cpu().numpy())
                    all_outputs_prob.extend(torch.sigmoid(labels_pre).cpu().numpy())
                val_loss /= len(val_loader.dataset)
            
            history['val_loss'].append(val_loss)
            all_labels = np.array(all_labels).flatten()
            all_outputs_prob = np.array(all_outputs_prob).flatten()
            all_outputs_binary = (all_outputs_prob > 0.5).astype(int)

            metrics = print_metrics(all_labels, all_outputs_prob, all_outputs_binary, phase=f"Epoch {epoch+1}/{num_epochs} Val")
            history['val_auc'].append(metrics['auc'])
            history['val_accuracy'].append(metrics['accuracy'])
            history['val_precision'].append(metrics["precision"])

            postfix_stats = {
                'train_loss': f"{train_loss:.4f}",
                'val_loss': f"{val_loss:.4f}",
                'val_auc': f"{metrics['auc']:.4f}",
                'val_acc': f"{metrics['accuracy']:.4f}"
            }
            
            epoch_iterator.set_postfix(postfix_stats)

            if metrics['auc'] > best_val_auc:
                best_val_auc = metrics['auc']
                epochs_no_improve = 0
                self.model.save(path=save_path)
                tqdm.write(f"Epoch {epoch+1}: New best model saved with val_loss: {best_val_auc:.4f}") # Use tqdm.write for this
            else:
                epochs_no_improve += 1
                if epochs_no_improve > patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break
        
        print("Training complete.")
        return history

    def predict(self, data_loader:DataLoader):
        self.model.eval()
        predictions_prob = []
        with torch.no_grad():
            for features in data_loader:
                outputs = self.model(features)
                predictions_prob.extend(torch.sigmoid(outputs).cpu().numpy())
        return np.array(predictions_prob).flatten()

def train_adaboost(model, X:pd.DataFrame, y:pd.Series, skf) -> None:
    X = X.reset_index(drop=True).values
    y = y.reset_index(drop=True)
    auc_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model.fit(X_train, y_train)
        
        y_prob = model.predict_proba(X_val)[:, 1]
        binary = (y_prob > 0.5).astype(int)
        auc = roc_auc_score(y_val, y_prob)
        print_metrics(y_val, y_prob, binary)

        auc_scores.append(auc)
        print(f"Fold {fold}: AUC = {auc}")

    print(f"Average AUC: {np.mean(auc_scores)}\n")

@hydra.main(config_name="train", config_path="config", version_base="1.3")
def train(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_df = pd.read_csv(os.path.join(config.data.path, config.data.train_file), index_col='customer_id')

    X = train_df.drop("label", axis=1).reset_index(drop=True)
    y = train_df["label"].reset_index(drop=True)

    # train_adaboost(copy.deepcopy(X), copy.deepcopy(y))
    # return

    X = X.replace([np.inf, -np.inf], np.nan)
    for col in X.columns:
        if X[col].isnull().any():
            print(f"Warning: Column '{col}' contains NaN values. Imputing with mean.")
            X[col] = X[col].fillna(X[col].mean())

    X_train_df, X_val_df, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=config.train.seed, stratify=y)

    train_dataset = Dataset(X_train_df, y_train, device=device)
    val_dataset = Dataset(X_val_df, y_val, device=device)
    
    train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.train.batch_size, shuffle=False)

    trainer = Trainer(device=device)
    trainer.train(train_loader=train_loader, val_loader=val_loader, load_path=config.model.load_path, save_path=config.model.save_path, num_epochs=config.train.num_epochs, patience=config.train.patience)

    test_df = pd.read_csv(os.path.join(config.data.path, config.data.test_file), index_col='customer_id')
    sub_df = pd.read_csv(os.path.join(config.submit.path, config.submit.file))
    test_dataset = Dataset(test_df, device=device)
    test_loader = DataLoader(test_dataset, batch_size=config.train.batch_size, shuffle=False)
    pred = trainer.predict(test_loader)
    pred_df = pd.DataFrame({"customer_id": test_df.index, "pred": pred})

    submission = pd.merge(sub_df, pred_df, on='customer_id', how='left')
    submission.fillna(0,inplace=True)
    submission = submission[['customer_id','pred']]
    submission.rename(columns={'customer_id':'customer_id','pred':'result'}, inplace=True)
    submission.loc[submission['result']>0.5, 'result'] = 1
    submission.loc[submission['result']<=0.5, 'result'] = 0
    submission.to_csv('submission.csv', index=False)

if __name__ == "__main__":
    train()