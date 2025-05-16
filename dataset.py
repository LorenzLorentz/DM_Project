import numpy as np
import pandas as pd
import torch
import torch.utils.data as data

class Dataset(data.Dataset):
    def __init__(self, features, labels=None, scaler=None, fit_scaler=False):
        """
        Args:
            features (pd.DataFrame or np.ndarray): Input features.
            labels (pd.Series or np.ndarray, optional): Target labels.
            scaler (StandardScaler, optional): Pre-fitted scaler.
            fit_scaler (bool): If True and scaler is provided, fit the scaler.
                               If True and scaler is None, create and fit a new one.
        """
        if isinstance(features, pd.DataFrame):
            features = features.values.astype(np.float32)

        if scaler:
            if fit_scaler:
                self.features = scaler.fit_transform(features).astype(np.float32)
            else:
                self.features = scaler.transform(features).astype(np.float32)
            self.scaler = scaler
        else:
            self.features = features.astype(np.float32)
            self.scaler = None


        self.labels = labels
        if self.labels is not None:
            if isinstance(labels, pd.Series):
                self.labels = labels.values.astype(np.float32)
            self.labels = self.labels.reshape(-1, 1) # Ensure correct shape for BCEWithLogitsLoss

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature_vector = self.features[idx]
        if self.labels is not None:
            label = self.labels[idx]
            return torch.tensor(feature_vector, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
        else:
            return torch.tensor(feature_vector, dtype=torch.float32)

    def get_scaler(self):
        return self.scaler