import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import pandas as pd

import pickle
from tqdm import tqdm

from model import BaseModel
from dataset import Dataset
from utils import get_metric_func

class MLPNet(nn.Module):
    def __init__(self, input_dim:int=66, hidden_dims:list[int]=[128, 128, 64], output_dim:int=1, dropout_rate:float=0.2, device:str=None):
        super(MLPNet, self).__init__()
        self.device = device
        
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim)) # Batch norm for stability
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))

        self.network = nn.Sequential(*layers).to(device)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
            elif isinstance(m, nn.Embedding):
                with torch.no_grad():
                    one_hot = torch.eye(m.num_embeddings, m.embedding_dim)
                    noise = torch.randn_like(one_hot) * 0.1
                    m.weight.copy_(one_hot + noise)

    def save(self, path:str, name:str=None):
        if name:
            torch.save(self.state_dict(), f"{path}/{name}.pth")
        else:
            time=datetime.datetime.now().strftime("%m%d%H%M")
            torch.save(self.state_dict(), f"{path}/MLPModel_{time}.pth")

    def load(self, path:str):
        if path:
            self.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        else:
            self.init_weight()
            self.to(self.device)

class MLPModel(BaseModel):
    def __init__(
        self, if_predict:bool=False, lr:float=None, num_epochs:int=None,load_path:str=None, save_path:str=None, random_seed:int=3407,
        input_dim:int=66, hidden_dims:list[int]=[128, 128, 64], output_dim:int=1, dropout_rate:float=0.2, criterion:str=None, batch_size:int=None, device:str=None, 
    ):
        super().__init__()
        self.num_epochs = num_epochs
        self.device = device
        self.save_path = save_path
        self.load_path = load_path
        self.if_predict = if_predict
        self.batch_size = batch_size

        self.net = MLPNet(input_dim, hidden_dims, output_dim, dropout_rate, device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.load()

        if criterion=="MSE":
            self.criterion = nn.MSELoss()
        elif criterion=="BCEWithLogits":
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError()

    def load(self):
        if self.load_path:
            self.net.load(self.load_path)
        elif self.if_predict:
            raise ValueError("Model path is needed for prediction")
        else:
            self.net.init_weight()

    def save(self):
        self.net.save(self.save_path)

    def train(
        self, feature:pd.DataFrame, label:pd.Series, eval:list[tuple[pd.DataFrame, pd.Series]], eval_metrics:list[str], verbose:int,
    ):
        if isinstance(eval, list):
            eval = eval[0]

        train_dataset = Dataset(feature, label, device=self.device)
        eval_dataset = Dataset(eval[0], eval[1], device=self.device)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        eval_loader = DataLoader(eval_dataset, batch_size=self.batch_size, shuffle=False)

        epoch_iterator = tqdm(range(self.num_epochs), desc="Epochs")
        for epoch in epoch_iterator:
            batch_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} Train", leave=False)
            self.net.train()
            for features, labels in batch_iterator:
                labels_pre = self.net(features)
                loss = self.criterion.forward(labels_pre, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            if (epoch+1) % verbose == 0:
                self.net.eval()
                eval_batch_iterator = tqdm(eval_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} Validation", leave=False)
                with torch.no_grad():
                    metric_results = {}
                    metric_funcs = {}

                    all_pre = []
                    all_label = []

                    for eval_metric in eval_metrics:
                        metric_funcs[eval_metric] = metric_funcs.get(eval_metric, get_metric_func(eval_metric))
                        metric_results[eval_metric] = metric_results.get(eval_metric, [])

                    for features, labels in eval_batch_iterator:
                        labels_pre = self.net.forward(features)
                        all_pre.extend(labels_pre.detach().cpu().numpy())
                        all_label.extend(labels)
                    
                    for eval_metric in eval_metrics:
                        metric_results[eval_metric] = metric_funcs[eval_metric](all_label, all_pre)

                    print(f"Epoch {epoch+1}:")
                    for eval_metric in eval_metrics:
                        print(f"eval metric {eval_metric} = {metric_results[eval_metric]}, ")
        self.save()

    def predict(self, feature:pd.DataFrame) -> pd.Series:
        with torch.no_grad:
            return self.net.forward(feature)