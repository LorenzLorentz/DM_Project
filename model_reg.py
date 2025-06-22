import pandas as pd
import numpy as np
import pickle
import datetime
from tqdm import tqdm

from model import BaseModel
from utils import get_metric_func

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1/(1+np.exp(-z))

class RegNet:
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, 1))

        for _ in range(self.n_iterations):
            linear_model = np.dot(X, self.weights)
            predictions = _sigmoid(linear_model)
            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            self.weights -= self.learning_rate * dw

    def predict_prob(self, X:np.ndarray) -> np.ndarray:
        linear_model = np.dot(X, self.weights)
        return _sigmoid(linear_model)

    def predict(self, X:np.ndarray, threshold:float=0.5) -> np.ndarray:
        return (self.predict_prob(X) >= threshold).astype(int)

class RegModel(BaseModel):
    def __init__(self, lr:float=0.1, num_epochs:int=3000, load_path:str=None, save_path:str=None, random_seed:int=3407):
        super().__init__(lr, num_epochs, load_path, save_path, random_seed)

        self.save_path = save_path
        self.model = RegNet(learning_rate=lr, n_iterations=num_epochs)
        self.X_mean = None
        self.X_std = None

        if random_seed is not None:
            np.random.seed(random_seed)
            
        if load_path:
            self.load(load_path)

    def preprocess(self, feature: pd.DataFrame, is_training: bool = False) -> np.ndarray:
        X = feature.values
        if is_training:
            self.X_mean = np.mean(X, axis=0)
            self.X_std = np.std(X, axis=0)
        
        self.X_std[self.X_std == 0] = 1

        X_standardized = (X - self.X_mean) / self.X_std
        X_processed = np.hstack([np.ones((X_standardized.shape[0], 1)), X_standardized])
        return X_processed

    def train(self, feature: pd.DataFrame, label: pd.Series, eval: list[tuple[pd.DataFrame, pd.Series]]=None, eval_metrics: list[str]=None, verbose:int=1):
        X_train = self.preprocess(feature, is_training=True)
        y_train = label.values.reshape(-1, 1)
        
        n_samples, n_features = X_train.shape
        if self.model.weights is None:
            self.model.weights = np.zeros((n_features, 1))

        epoch_iterator = tqdm(range(self.model.n_iterations), desc="Epochs")
        for i in epoch_iterator:
            linear_model = np.dot(X_train, self.model.weights)
            predictions = _sigmoid(linear_model)
            dw = (1/n_samples) * np.dot(X_train.T,(predictions-y_train))
            self.model.weights -= self.model.learning_rate * dw

            if verbose > 0 and (i + 1) % verbose == 0:
                print(f"Iteration {i+1}/{self.model.n_iterations}:")
                if eval and eval_metrics:
                    eval_feature, eval_label = eval[0]
                    X_eval = self.preprocess(eval_feature, is_training=False)
                    eval_predictions = self.model.predict_prob(X_eval)
                    for metric_name in eval_metrics:
                        metric_func = get_metric_func(metric_name)
                        score = metric_func(eval_label, eval_predictions)
                        print(f"  eval metric {metric_name} = {score:.6f}")
        
        if self.save_path:
            time=datetime.datetime.now().strftime("%m%d%H%M")
            self.save(f"{self.save_path}/RegModel_{time}.pth")

    def predict(self, feature: pd.DataFrame) -> pd.Series:
        X_pred = self._preprocess_features(feature, is_training=False)
        predictions = self.model.predict_prob(X_pred)
        return pd.Series(predictions.flatten(), index=feature.index, name='prediction')

    def save(self, path: str):
        with open(path, 'wb') as f:
            state = {
                'model': self.model,
                'mean': self.X_mean,
                'std': self.X_std
            }
            pickle.dump(state, f)
    
    def load(self, path: str):
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.model = state['model']
        self.X_mean = state['mean']
        self.X_std = state['std']
        print(f"Model loaded from {path}")