import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, f1_score
from tqdm import tqdm
from typing import Dict, List

from model import BaseModel

class AttentionLayer(nn.Module):
    def __init__(self, embedding_dim):
        super(AttentionLayer, self).__init__()
        # 注意力网络，输入维度是 4*embedding_dim（拼接了 q, k, q-k, q*k）
        self.attention_net = nn.Sequential(
            nn.Linear(embedding_dim * 4, 80),
            nn.ReLU(),
            nn.Linear(80, 1),
        )

    def forward(self, query, keys, keys_mask):
        # query: (B, 1, D) - 候选物品 embedding
        # keys: (B, T, D) - 历史行为 embedding
        # keys_mask: (B, T) - 标记哪些是 padding
        
        batch_size, seq_len, dim = keys.shape
        
        # 将 query 扩展以匹配 keys 的维度
        query = query.repeat(1, seq_len, 1)  # (B, T, D)
        
        # 将 query, keys, 差，积拼接，构造更丰富的交叉特征
        info = torch.cat([query, keys, query - keys, query * keys], dim=-1) # (B, T, 4*D)
        
        # 计算注意力得分
        scores = self.attention_net(info).squeeze(-1)  # (B, T)
        
        # 应用 mask，将 padding 位置的得分设为极小的数，使其在 softmax 后权重接近 0
        scores = scores.masked_fill(keys_mask == 0, -1e9)
        
        # Softmax 归一化得到权重
        weights = torch.softmax(scores, dim=1)  # (B, T)
        
        # 加权求和得到用户兴趣的向量表示
        # weights需要扩展维度 (B, T) -> (B, T, 1) 以便与 keys (B, T, D) 相乘
        weighted_sum = (weights.unsqueeze(-1) * keys).sum(dim=1)  # (B, D)
        
        return weighted_sum

class DIN(nn.Module):
    def __init__(self, num_users, num_items, num_categories, embedding_dim=32):
        # ... (初始化代码与原来相同) ...
        super(DIN, self).__init__()
        self.embedding_dim = embedding_dim
        
        # Embedding 层 (词汇表大小应为 原始数量 + 1 以容纳 padding_idx=0)
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        self.cat_embedding = nn.Embedding(num_categories, embedding_dim, padding_idx=0)
        
        # 注意力层
        # 输入维度是 item_emb 和 cat_emb 拼接后的维度，即 2 * embedding_dim
        self.attention = AttentionLayer(embedding_dim * 2) # AttentionLayer 的输入是 4*D，D是keys的维度
        
        # MLP 层
        # 输入维度: user_emb + attention_output + candidate_emb
        # D_user + D_attention + D_candidate = D + 2*D + 2*D = 5*D
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 5, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, data: dict):
        # --- 修正: 使用与 DataLoader 一致的键名 ---
        user_id = data['customer_id']
        candidate_good = data['candidate_good']
        candidate_class = data['candidate_class']
        hist_goods = data['history_goods']
        hist_classes = data['history_classes']
        
        # --- Embedding ---
        # 用户的 Embedding
        user_emb = self.user_embedding(user_id) # (B, D)
        
        # 候选物品的 Embedding (物品和品类拼接)
        candidate_good_emb = self.item_embedding(candidate_good) # (B, D)
        candidate_class_emb = self.cat_embedding(candidate_class) # (B, D)
        candidate_emb = torch.cat([candidate_good_emb, candidate_class_emb], dim=-1) # (B, 2*D)
        
        # 历史行为序列的 Embedding (物品和品类拼接)
        hist_goods_emb = self.item_embedding(hist_goods) # (B, T, D)
        hist_classes_emb = self.cat_embedding(hist_classes) # (B, T, D)
        hist_emb = torch.cat([hist_goods_emb, hist_classes_emb], dim=-1) # (B, T, 2*D)
        
        # --- 注意力机制 ---
        # 创建 mask: 历史记录中不为 0 (padding) 的位置为 True
        mask = (hist_goods != 0) # (B, T)
        
        # attention_output 是用户对历史行为序列的加权兴趣表示
        # 注意力层输入 query 的维度需要是 (B, 1, D_key)
        attention_output = self.attention(candidate_emb.unsqueeze(1), hist_emb, mask) # (B, 2*D)

        # --- MLP ---
        # 拼接所有特征向量作为全连接层的输入
        combined_vec = torch.cat([user_emb, candidate_emb, attention_output], dim=-1) # (B, 5*D)
        
        # 预测，输出一个 logit 值，后续会通过 Sigmoid 转换为概率
        output = self.mlp(combined_vec).squeeze(-1) # (B,)
        
        return output

class DINModel(BaseModel):
    def __init__(
        self, lr:float=1e-3, num_epochs:int=10, load_path:str=None, save_path:str=None, random_seed:int=42,
        num_users:int=None, num_items:int=None, num_categories:int=None,
        batch_size:int=128, embedding_dim:int=32, criterion:str="bce", device:str="cuda" if torch.cuda.is_available() else "cpu", 
    ):
        super().__init__(lr, num_epochs, load_path, save_path, random_seed)
        self.lr = lr
        self.load_path = load_path
        self.num_epochs = num_epochs
        self.save_path = save_path
        self.device = torch.device(device)

        self.net = DIN(
            num_users=num_users,
            num_items=num_items,
            num_categories=num_categories,
            embedding_dim=embedding_dim
        ).to(self.device)

        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        
        if criterion.lower() == "bce":
            self.criterion = nn.BCEWithLogitsLoss()
        elif criterion.lower() == "mse":
            self.criterion = nn.MSELoss()
        else:
            raise NotImplementedError()

        if self.load_path:
            self.net.load_state_dict(torch.load(self.load_path, map_location=self.device))
            print(f"Model loaded from {self.load_path}")

    def loss(self, features:Dict[str, torch.Tensor], labels:torch.Tensor) -> torch.Tensor:
        logits = self.net(features)
        return self.criterion(logits, labels)

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader, desc="Evaluation"):
        self.net.eval()
        
        total_loss = 0.0
        all_labels = []
        all_preds = []

        progress_bar = tqdm(data_loader, desc=desc, leave=False)
        for batch in progress_bar:
            features = {k: v.to(self.device) for k, v in batch.items() if k != 'label'}
            labels = batch['label'].to(self.device)

            logits = self.net(features)
            loss = self.criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)

            all_labels.append(labels.cpu())
            all_preds.append(probs.cpu())

        all_labels = torch.cat(all_labels).numpy()
        all_preds = torch.cat(all_preds).numpy()

        avg_loss = total_loss / len(data_loader)
        auc = roc_auc_score(all_labels, all_preds)

        binary_preds = (all_preds >= 0.5).astype(int)
        accuracy = accuracy_score(all_labels, binary_preds)
        precision = precision_score(all_labels, binary_preds)
        f1 = f1_score(all_labels, binary_preds)
        
        metrics = {
            "loss": avg_loss,
            "auc": auc,
            "accuracy": accuracy,
            "precision": precision,
            "f1_score": f1
        }

        print(metrics)
        
        return metrics

    def train(self, train_loader:DataLoader, val_loader:DataLoader=None):
        print("* Start Training...")
        best_val_auc = 0.0

        for epoch in range(self.num_epochs):
            self.net.train()
            total_train_loss = 0.0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Training]", leave=True)
            for batch in progress_bar:
                features = {k: v.to(self.device) for k, v in batch.items() if k != 'label'}
                labels = batch['label'].to(self.device)
                
                self.optimizer.zero_grad()
                loss = self.loss(features, labels)
                loss.backward()
                self.optimizer.step()
                
                total_train_loss += loss.item()
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            avg_train_loss = total_train_loss/len(train_loader)

            log_message = f"Epoch {epoch+1}/{self.num_epochs} | Avg Training Loss: {avg_train_loss:.4f}"

            if val_loader:
                val_metrics = self.evaluate(val_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Evaluation]")

                log_message += f" | Validation Set Loss: {val_metrics['loss']:.4f}" \
                               f" | AUC: {val_metrics['auc']:.4f}" \
                               f" | Accuracy: {val_metrics['accuracy']:.4f}" \
                               f" | Precision: {val_metrics['precision']:.4f}" \
                               f" | F1_Score: {val_metrics['f1_score']:.4f}" 
                
                if val_metrics['auc'] > best_val_auc:
                    best_val_auc = val_metrics['auc']
                    log_message += "* New BEST!"
                    if self.save_path:
                        torch.save(self.net.state_dict(), self.save_path)
            
            print(log_message)
        
        print("\n* Training Completed!")
        if self.save_path and best_val_auc > 0:
             print(f"Best Model (AUC: {best_val_auc:.4f}) has been save to: {self.save_path}")

    @torch.no_grad()
    def predict(self, data_loader:DataLoader) -> List[float]:
        self.net.eval()
        predictions = []
        for batch in tqdm(data_loader):
            features = {k: v.to(self.device) for k, v in batch.items() if k != 'label'}
            logits = self.net(features)
            probs = torch.sigmoid(logits).cpu().numpy().tolist()
            predictions.extend(probs)
        return predictions