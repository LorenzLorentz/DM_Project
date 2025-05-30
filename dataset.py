import numpy as np
import pandas as pd
import torch
import torch.utils.data as data

class Dataset(data.Dataset):
    def __init__(self, features, labels=None, device:str=None):
        """
        Args:
            features (pd.DataFrame or np.ndarray): Input features.
            labels (pd.Series or np.ndarray, optional): Target labels.
            scaler (StandardScaler, optional): Pre-fitted scaler.
            fit_scaler (bool): If True and scaler is provided, fit the scaler.
                               If True and scaler is None, create and fit a new one.
        """
        self.device = device

        if isinstance(features, pd.DataFrame):
            features = features.values.astype(np.float32)

        self.features = features
        self.labels = labels
        
        if self.labels is not None:
            if isinstance(labels, pd.Series):
                self.labels = labels.values.astype(np.float32)
            self.labels = self.labels.reshape(-1, 1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature_vector = self.features[idx]
        if self.labels is not None:
            label = self.labels[idx]
            return torch.tensor(feature_vector, dtype=torch.float32, device=self.device), torch.tensor(label, dtype=torch.float32, device=self.device)
        else:
            return torch.tensor(feature_vector, dtype=torch.float32, device=self.device)

    def get_scaler(self):
        return self.scaler
    
class DinDataset(Dataset):
    def __init__(self, dataframe:pd.DataFrame):
        self.data = dataframe

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx:int) -> dict[str, torch.Tensor]:
        row = self.data.iloc[idx]

        return {
            'customer_id': torch.tensor(row['customer_id'], dtype=torch.long),
            'history_goods': torch.tensor(row['history_goods'], dtype=torch.long),
            'history_classes': torch.tensor(row['history_classes'], dtype=torch.long),
            'candidate_good': torch.tensor(row['candidate_good'], dtype=torch.long),
            'candidate_class': torch.tensor(row['candidate_class'], dtype=torch.long),
            'label': torch.tensor(row['label'], dtype=torch.float32)
        }