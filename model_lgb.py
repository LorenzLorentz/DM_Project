import os
import datetime

from model import BaseModel

import pandas as pd

import lightgbm as lgb
import joblib

class LGBModel(BaseModel):
    def __init__(
        self, if_predict:bool=False, lr:float=None, num_epochs:int=None, load_path:str=None, save_path:str=None, random_seed:int=None,
        num_leaves:int=31, reg_alpha:float=0.25, reg_lambda:float=0.25, max_depth:int=-1, min_child_samples:int=3, subsample:int=1, colsample_bytree:int=1
    ):
        super().__init__()
        self.load_path = load_path
        self.save_path = save_path

        if load_path:
            self.clf = joblib.load(load_path)
        elif if_predict:
            raise ValueError("Model path is needed for prediction")
        else:
            self.clf = lgb.LGBMClassifier(
                num_leaves=num_leaves, reg_alpha=reg_alpha, reg_lambda=reg_lambda, objective='binary',
                max_depth=max_depth, learning_rate=lr, min_child_samples=min_child_samples, random_state=random_seed,
                n_estimators=num_epochs, subsample=subsample, colsample_bytree=colsample_bytree,
            )

    def train(self, feature:pd.DataFrame, label:pd.Series, eval:tuple[pd.DataFrame, pd.Series], eval_metrics:list[str], verbose:int):
        self.clf.fit(feature, label, eval_set=eval, eval_metric=eval_metrics, callbacks=[lgb.log_evaluation(period=verbose)])
        time=datetime.datetime.now().strftime("%m%d%H%M")
        joblib.dump(self.clf, os.path.join(self.save_path, f"LGBModel_{time}.joblib")) 
    
    def predict(self, feature:pd.DataFrame):
        return self.clf.predict_proba(feature)
    
    def load(self):
        self.clf = joblib.load(self.load_path)