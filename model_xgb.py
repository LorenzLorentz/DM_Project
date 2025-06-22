import os

from model import BaseModel

import xgboost as xgb

import pandas as pd

import datetime

class XGBModel(BaseModel):
    def __init__(
        self, if_predict:bool=False, lr:float=None, num_epochs:int=None, load_path:str=None, save_path:str=None, random_seed:int=3407,
        max_depth:int=5, subsample:int=1, colsample_bytree:int=1, reg_alpha:float=0.25, reg_lambda:float=0.25, min_child_weight:int=3, use_label_encoder:bool=False
    ):
        super().__init__()
        self.load_path = load_path
        self.save_path = save_path
        
        self.clf = xgb.XGBClassifier(
            max_depth=max_depth,
            learning_rate=lr,
            n_estimators=num_epochs,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            objective='binary:logistic',
            random_state=random_seed,
            min_child_weight=min_child_weight,
            use_label_encoder=use_label_encoder,
            eval_metric='auc'
        )

        if load_path:
            self.clf.load_model(load_path)
        elif if_predict:
            raise ValueError("Model path is needed for prediction")

    def train(self, feature:pd.DataFrame, label:pd.Series, eval:tuple[pd.DataFrame, pd.Series], eval_metrics:list[str], verbose:int):
        self.clf.fit(feature, label, eval_set=eval, verbose=verbose)
        time=datetime.datetime.now().strftime("%m%d%H%M")
        self.clf.save_model(os.path.join(self.save_path, f"XGBModel_{time}.json"))
    
    def predict(self, feature:pd.DataFrame) -> pd.Series:
        return self.clf.predict_proba(feature)

    def load(self):
        self.clf.load_model(self.load_path)