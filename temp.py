import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# --- 0. Configuration & Helper Functions ---
DATA_PATH = './' # Directory where your processed CSVs are stored
TRAIN_CSV = 'X_train_processed.csv'
TEST_CSV = 'X_test_processed.csv' # Assuming this is for prediction/final eval
LABEL_COLUMN = 'label'
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# For reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def print_metrics(y_true, y_pred_proba, y_pred_binary, phase="Validation"):
    """Prints common classification metrics."""
    auc = roc_auc_score(y_true, y_pred_proba)
    acc = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    print(f"{phase} Metrics: AUC={auc:.4f} | Acc={acc:.4f} | Precision={precision:.4f} | Recall={recall:.4f} | F1={f1:.4f}")
    return {"auc": auc, "accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# --- 1. Dataset Class (PyTorch) ---
class TabularDataset(Dataset):
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

# --- 2. Model Class (PyTorch MLP) ---
class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64], output_dim=1, dropout_rate=0.3):
        super(MLPModel, self).__init__()
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim)) # Batch norm for stability
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        # No sigmoid here, as BCEWithLogitsLoss is more numerically stable

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# --- 3. Trainer Function (PyTorch) ---
def train_model_pytorch(model, train_loader, val_loader, criterion, optimizer, num_epochs=20, device=DEVICE, patience=5):
    """
    A basic training loop for a PyTorch model.
    Includes simple early stopping.
    """
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'val_auc': [], 'val_accuracy': []}

    print(f"Training on {device}...")
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_train_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * features.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_train_loss)

        # Validation phase
        model.eval()  # Set model to evaluation mode
        running_val_loss = 0.0
        all_labels = []
        all_outputs_proba = []

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * features.size(0)

                all_labels.extend(labels.cpu().numpy())
                all_outputs_proba.extend(torch.sigmoid(outputs).cpu().numpy()) # Apply sigmoid for probabilities

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        history['val_loss'].append(epoch_val_loss)

        all_labels = np.array(all_labels).flatten()
        all_outputs_proba = np.array(all_outputs_proba).flatten()
        all_outputs_binary = (all_outputs_proba > 0.5).astype(int)

        metrics = print_metrics(all_labels, all_outputs_proba, all_outputs_binary, phase=f"Epoch {epoch+1}/{num_epochs} Val")
        history['val_auc'].append(metrics['auc'])
        history['val_accuracy'].append(metrics['accuracy'])


        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

        # Early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            # Save the best model (optional)
            # torch.save(model.state_dict(), 'best_model_pytorch.pth')
            # print("Best model saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
    
    print("Training complete.")
    return history

def predict_pytorch(model, data_loader, device=DEVICE):
    model.eval()
    predictions_proba = []
    with torch.no_grad():
        for features in data_loader:
            if isinstance(features, list): # if DataLoader returns (features, None) for test set
                features = features[0]
            features = features.to(device)
            outputs = model(features)
            predictions_proba.extend(torch.sigmoid(outputs).cpu().numpy())
    return np.array(predictions_proba).flatten()

# --- 4. Main Execution Block ---
if __name__ == '__main__':
    # --- PyTorch Workflow ---
    print("Starting PyTorch Workflow...")
    try:
        train_val_df = pd.read_csv(os.path.join(DATA_PATH, TRAIN_CSV), index_col='customer_id')
    except FileNotFoundError:
        print(f"Error: {TRAIN_CSV} not found in {DATA_PATH}. Please run the data processing script first.")
        exit()

    if LABEL_COLUMN not in train_val_df.columns:
        print(f"Error: Label column '{LABEL_COLUMN}' not found in {TRAIN_CSV}.")
        exit()

    X = train_val_df.drop(LABEL_COLUMN, axis=1)
    y = train_val_df[LABEL_COLUMN]

    # Handle potential NaN/inf values that might have slipped through or resulted from operations
    X = X.replace([np.inf, -np.inf], np.nan)
    # Simple NaN imputation strategy: fill with mean. More sophisticated methods might be needed.
    for col in X.columns:
        if X[col].isnull().any():
            print(f"Warning: Column '{col}' contains NaN values. Imputing with mean.")
            X[col] = X[col].fillna(X[col].mean())


    X_train_df, X_val_df, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

    # Initialize scaler
    scaler = StandardScaler()

    # Create Datasets and DataLoaders
    train_dataset = TabularDataset(X_train_df, y_train, scaler=scaler, fit_scaler=True)
    # Get the fitted scaler from the training set to apply to validation and test sets
    fitted_scaler = train_dataset.get_scaler()
    val_dataset = TabularDataset(X_val_df, y_val, scaler=fitted_scaler, fit_scaler=False)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    # Instantiate model, loss, optimizer
    input_dim = X_train_df.shape[1]
    pytorch_model = MLPModel(input_dim=input_dim).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss() # Handles sigmoid internally, good for binary classification
    optimizer = optim.AdamW(pytorch_model.parameters(), lr=0.001, weight_decay=0.01)

    # Train the model
    train_history_pytorch = train_model_pytorch(pytorch_model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device=DEVICE, patience=10)

    # --- (Optional) Prediction on Test Set (PyTorch) ---
    try:
        test_df_pytorch = pd.read_csv(os.path.join(DATA_PATH, TEST_CSV), index_col='customer_id')
        test_df_pytorch = test_df_pytorch.replace([np.inf, -np.inf], np.nan)
        for col in test_df_pytorch.columns:
            if test_df_pytorch[col].isnull().any():
                 test_df_pytorch[col] = test_df_pytorch[col].fillna(X[col].mean()) # Impute with train mean

        test_dataset_pytorch = TabularDataset(test_df_pytorch, labels=None, scaler=fitted_scaler, fit_scaler=False)
        test_loader_pytorch = DataLoader(test_dataset_pytorch, batch_size=256, shuffle=False)
        
        test_predictions_pytorch = predict_pytorch(pytorch_model, test_loader_pytorch, device=DEVICE)
        print("\nPyTorch Test Set Predictions (Probabilities):")
        # print(test_predictions_pytorch[:10]) # Print first 10 predictions
        
        # Create a submission file (example)
        submission_pytorch_df = pd.DataFrame({'customer_id': test_df_pytorch.index, 'prediction_pytorch': test_predictions_pytorch})
        submission_pytorch_df.to_csv('submission_pytorch.csv', index=False)
        print("PyTorch submission_pytorch.csv saved.")

    except FileNotFoundError:
        print(f"\n{TEST_CSV} not found. Skipping PyTorch test set prediction.")
    except Exception as e:
        print(f"\nError during PyTorch test set prediction: {e}")


    # --- Alternative: LightGBM Workflow ---
    print("\n\nStarting LightGBM Workflow...")
    try:
        import lightgbm as lgb

        # Data for LightGBM (can use the same splits, but LightGBM handles NaNs natively better)
        # For LightGBM, scaling is not strictly necessary and sometimes not beneficial.
        # We'll use the original unscaled X_train_df, X_val_df for this example.
        X_lgb = train_val_df.drop(LABEL_COLUMN, axis=1)
        y_lgb = train_val_df[LABEL_COLUMN]
        
        # Handle potential NaN/inf values (LightGBM can handle NaNs, but not Infs)
        X_lgb = X_lgb.replace([np.inf, -np.inf], np.nan)
        # LightGBM can often handle NaNs internally, but explicit imputation can also be done.
        # For this example, we'll let LightGBM handle NaNs if present, after Infs are removed.

        X_train_lgb, X_val_lgb, y_train_lgb, y_val_lgb = train_test_split(
            X_lgb, y_lgb, test_size=0.2, random_state=SEED, stratify=y_lgb
        )

        lgb_train = lgb.Dataset(X_train_lgb, y_train_lgb)
        lgb_val = lgb.Dataset(X_val_lgb, y_val_lgb, reference=lgb_train)

        params = {
            'objective': 'binary',
            'metric': ['auc', 'binary_logloss'],
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1, # Suppress GBDT messages
            'n_jobs': -1,
            'seed': SEED,
            'early_stopping_round': 50 # Renamed from early_stopping_rounds in newer versions
        }

        print("Training LightGBM model...")
        gbm_model = lgb.train(params,
                              lgb_train,
                              num_boost_round=1000, # Max rounds
                              valid_sets=[lgb_train, lgb_val],
                              callbacks=[lgb.log_evaluation(period=100)]) # Renamed from valid_sets/evals_result

        # Evaluate LightGBM model
        y_val_pred_proba_lgb = gbm_model.predict(X_val_lgb, num_iteration=gbm_model.best_iteration)
        y_val_pred_binary_lgb = (y_val_pred_proba_lgb > 0.5).astype(int)
        print("\nLightGBM Model Evaluation:")
        print_metrics(y_val_lgb, y_val_pred_proba_lgb, y_val_pred_binary_lgb, phase="LightGBM Validation")

        # --- (Optional) Prediction on Test Set (LightGBM) ---
        try:
            test_df_lgb = pd.read_csv(os.path.join(DATA_PATH, TEST_CSV), index_col='customer_id')
            test_df_lgb = test_df_lgb.replace([np.inf, -np.inf], np.nan) # Handle Infs

            test_predictions_lgb = gbm_model.predict(test_df_lgb, num_iteration=gbm_model.best_iteration)
            print("\nLightGBM Test Set Predictions (Probabilities):")
            # print(test_predictions_lgb[:10])

            submission_lgb_df = pd.DataFrame({'customer_id': test_df_lgb.index, 'prediction_lgb': test_predictions_lgb})
            submission_lgb_df.to_csv('submission_lgb.csv', index=False)
            print("LightGBM submission_lgb.csv saved.")

        except FileNotFoundError:
            print(f"\n{TEST_CSV} not found. Skipping LightGBM test set prediction.")
        except Exception as e:
            print(f"\nError during LightGBM test set prediction: {e}")


    except ImportError:
        print("\nLightGBM not installed. Skipping LightGBM workflow.")
    except Exception as e:
        print(f"\nAn error occurred in the LightGBM workflow: {e}")