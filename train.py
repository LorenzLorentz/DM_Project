import os
import hydra
import pickle
import torch

import pandas as pd

from model import BaseModel
from model_mlp import MLPModel
from model_xgb import XGBModel
from model_lgb import LGBModel
from model_reg import RegModel

def get_model(name:str, config, trained = False) -> BaseModel:
    if name == "mlp":
        with open(os.path.join(config.data.path, config.data.feature_method+config.data.train_file), 'rb') as f:
            train_df = pickle.load(f)
        return MLPModel(
            if_predict=config.mlp.if_predict, lr=config.mlp.lr, num_epochs=config.mlp.num_epochs, load_path=config.mlp.load_path, save_path=config.mlp.save_path, random_seed=config.train.random_seed,
            input_dim=(train_df.shape[1]-1), hidden_dims=config.mlp.hidden_dims, output_dim=1, dropout_rate=config.mlp.dropout_rate, criterion=config.mlp.criterion, batch_size=config.mlp.batch_size, device=("cuda" if torch.cuda.is_available() else "cpu"), 
        )
    elif name == "xgb":
        if trained:
            model = XGBModel(
                if_predict=config.xgb.if_predict, lr=config.xgb.lr, num_epochs=config.xgb.num_epochs, load_path=config.xgb.load_path, save_path=config.xgb.save_path, random_seed=config.train.random_seed,
                max_depth=config.xgb.max_depth, subsample=config.xgb.subsample, colsample_bytree=config.xgb.colsample_bytree, reg_alpha=config.xgb.reg_alpha, reg_lambda=config.xgb.reg_lambda, min_child_weight=config.xgb.min_child_weight, use_label_encoder=config.xgb.use_label_encoder,
            )
            model.load()
            return model
        else:
            return XGBModel(
                if_predict=config.xgb.if_predict, lr=config.xgb.lr, num_epochs=config.xgb.num_epochs, load_path=config.xgb.load_path, save_path=config.xgb.save_path, random_seed=config.train.random_seed,
                max_depth=config.xgb.max_depth, subsample=config.xgb.subsample, colsample_bytree=config.xgb.colsample_bytree, reg_alpha=config.xgb.reg_alpha, reg_lambda=config.xgb.reg_lambda, min_child_weight=config.xgb.min_child_weight, use_label_encoder=config.xgb.use_label_encoder,
            )
    elif name == "lgb":
        return LGBModel(
            if_predict=config.lgb.if_predict, lr=config.lgb.lr, num_epochs=config.lgb.num_epochs, load_path=config.lgb.load_path, save_path=config.lgb.save_path, random_seed=config.train.random_seed,
            num_leaves=config.lgb.num_leaves, reg_alpha=config.lgb.reg_alpha, reg_lambda=config.lgb.reg_lambda, max_depth=config.lgb.max_depth, min_child_samples=config.lgb.min_child_samples, subsample=config.lgb.subsample, colsample_bytree=config.lgb.colsample_bytree,
        )
    elif name == "reg":
        return RegModel(
            lr=config.reg.lr, num_epochs=config.reg.num_epochs, load_path=config.reg.load_path, save_path=config.reg.save_path, random_seed=config.train.random_seed,
        )
    else:
        raise NotImplementedError()

@hydra.main(config_name="train", config_path="config", version_base="1.3")
def train(config):    
    with open(os.path.join(config.data.path, config.data.feature_method+config.data.train_file), 'rb') as f:
        train_df = pickle.load(f)
    
    with open(os.path.join(config.data.path, config.data.feature_method+config.data.val_file), 'rb') as f:
        val_df = pickle.load(f)
    
    X_train = train_df.drop('label', axis=1)
    y_train = train_df['label']
    X_val = val_df.drop('label', axis=1)
    y_val = val_df['label']
    X_val = X_val[X_train.columns]

    model = get_model(name=config.model_name, config=config)
    model.train(
        feature=X_train,
        label=y_train,
        eval=[(X_val, y_val)],
        eval_metrics=['auc'],
        verbose=config.mlp.verbose,
    )

@hydra.main(config_name="train", config_path="config", version_base="1.3")
def predict(config):
    with open(os.path.join(config.data.path, config.data.feature_method+config.data.test_file), 'rb') as f:
        test_df = pickle.load(f)
    X_test = test_df

    model = get_model(name=config.model_name, config=config, trained=True)
    result = pd.DataFrame({'customer_id': test_df.index})
    result['result'] = model.predict(feature=X_test)[:, 1]
    result = result.sort_values("result", ascending=False).copy()

    buy_num = 450000
    result.index = range(len(result))
    result.loc[result.index <= buy_num, 'result'] = 1
    result.loc[result.index > buy_num, 'result'] = 0
    result['result'] = result['result'].astype(int)
    result.sort_values("customer_id", ascending=True, inplace=True)
    result.to_csv(f"submit/submission_{config.model_name}.csv", index=False)

if __name__ == "__main__":
    # train()
    predict()