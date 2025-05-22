import os
import hydra
import pickle

import pandas as pd

from model import BaseModel
from model_mlp import MLPModel
from model_xgb import XGBModel
from model_lgb import LGBModel

def get_model(name:str, config) -> BaseModel:
    if name == "mlp":
        return MLPModel(
            if_predict=config.mlp.if_predict, lr=config.mlp.lr, num_epochs=config.mlp.num_epochs, load_path=config.mlp.load_path, save_path=config.mlp.save_path, random_seed=config.train.random_seed,
            input_dim=config.mlp.input_dim, hidden_dims=config.mlp.hidden_dims, output_dim=1, dropout_rate=config.mlp.dropout_rate, criterion=config.mlp.criterion, batch_size=config.mlp.batch_size, device="cpu", 
        )
    elif name == "xgb":
        return XGBModel(
            if_predict=config.xgb.if_predict, lr=config.xgb.lr, num_epochs=config.xgb.num_epochs, load_path=config.xgb.load_path, save_path=config.xgb.save_path, random_seed=config.train.random_seed,
            max_depth=config.xgb.max_depth, subsample=config.xgb.subsample, colsample_bytree=config.xgb.colsample_bytree, reg_alpha=config.xgb.reg_alpha, reg_lambda=config.xgb.reg_lambda, min_child_weight=config.xgb.min_child_weight, use_label_encoder=config.xgb.use_label_encoder,
        )
    elif name == "lgb":
        return LGBModel(
            if_predict=config.lgb.if_predict, lr=config.lgb.lr, num_epochs=config.lgb.num_epochs, load_path=config.lgb.load_path, save_path=config.lgb.save_path, random_seed=config.train.random_seed,
            num_leaves=config.lgb.num_leaves, reg_alpha=config.lgb.reg_alpha, reg_lambda=config.lgb.reg_lambda, max_depth=config.lgb.max_depth, min_child_samples=config.lgb.min_child_samples, subsample=config.lgb.subsample, colsample_bytree=config.lgb.colsample_bytree,
        )
    else:
        raise NotImplementedError()

@hydra.main(config_name="train", config_path="config", version_base="1.3")
def train(config):
    with open(os.path.join(config.data.path, "train.pkl"), 'rb') as f:
        train = pickle.load(f)
    
    with open(os.path.join(config.data.path, "test.pkl"), 'rb') as f:
        test = pickle.load(f)

    train = train.reset_index()
    test = test.reset_index()
    all_df=pd.concat([train,test],axis=0)
    train = all_df[all_df['label'].notnull()]
    test = all_df[all_df['label'].isnull()]

    model = get_model(name=config.model_name, config=config)
    model.train(
        feature=train.drop(['label', 'customer_id'],axis=1),
        label=train['label'],
        eval=[(train.drop(['label', 'customer_id'],axis=1), train['label'])],
        eval_metrics=['auc'],
        verbose=config.mlp.verbose,
    )

@hydra.main(config_name="train", config_path="config", version_base="1.3")
def predict(config):
    with open(os.path.join(config.data.path, config.data.train_file), 'rb') as f:
        train = pickle.load(f)

    with open(os.path.join(config.data.path, config.data.test_file), 'rb') as f:
        test = pickle.load(f)

    train = train.reset_index()
    test = test.reset_index()
    all_df = pd.concat([train,test],axis=0)
    train = all_df[all_df['label'].notnull()]
    test = all_df[all_df['label'].isnull()]

    cols = train.columns.tolist()
    cols.remove("label")
    cols.remove('customer_id')

    model = get_model(name=config.model_name, config=config)
    model.predict()

    pred = model.predict(test.drop(["label", "customer_id"], axis=1))[:, 1] 
    result = pd.read_csv(os.path.join(config.submit.path, config.submit.file))
    result['result'] = pred
    result = result.sort_values("result", ascending=False).copy()

    buy_num = 450000
    result.index = range(len(result))
    result.loc[result.index <= buy_num, 'result'] = 1
    result.loc[result.index > buy_num, 'result'] = 0
    result.sort_values("customer_id", ascending=True, inplace=True)
    result.to_csv("submit/submission_{}.csv".format(config.model_name), index=False)

if __name__ == "__main__":
    train()