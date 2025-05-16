import torch
import torch.nn as nn
import torch.optim as optim

import datetime

class MLPModel(nn.Module):
    def __init__(self, input_dim:int, hidden_dims:list[int]=[128, 128, 64], output_dim:int=1, dropout_rate:float=0.2, device:str=None):
        super(MLPModel, self).__init__()
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

    def forward(self, x):
        return self.network(x)
    
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
            elif isinstance(m, nn.Embedding):
                with torch.no_grad():
                    one_hot = torch.eye(m.num_embeddings, m.embedding_dim)
                    noise = torch.randn_like(one_hot) * 0.1
                    m.weight.copy_(one_hot + noise)

    def save(self, path:str):
        time=datetime.datetime.now().strftime("%m%d%H%M")
        torch.save(self.state_dict(), f"{path}/MLPModel_{time}.pth")

    def load(self, path:str):
        if path:
            self.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        else:
            self.init_weight()
            self.to(self.device)