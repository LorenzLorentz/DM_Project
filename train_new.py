import os

import pickle
import lightgbm as lgb
import xgboost as xgb
import pandas as pd

def train_(data_path:str,model_path:str):
    with open(os.path.join(data_path, "train.pkl"), 'rb') as f:
        train = pickle.load(f)
    
    with open(os.path.join(data_path, "test.pkl"), 'rb') as f:
        test = pickle.load(f)

    train = train.reset_index()
    test = test.reset_index()
    
    all_df=pd.concat([train,test],axis=0)
    train = all_df[all_df['label'].notnull()]
    test = all_df[all_df['label'].isnull()]
    
    # clf = lgb.LGBMClassifier(
    #     num_leaves=2**5-1, reg_alpha=0.25, reg_lambda=0.25, objective='binary',
    #     max_depth=-1, learning_rate=0.005, min_child_samples=3, random_state=3407,
    #     n_estimators=2500, subsample=1, colsample_bytree=1,
    # )

    clf = xgb.XGBClassifier(
        max_depth=5,
        learning_rate=0.005,
        n_estimators=2500,
        subsample=1,
        colsample_bytree=1,
        reg_alpha=0.25,
        reg_lambda=0.25,
        objective='binary:logistic',
        random_state=3407,
        min_child_weight=3,
        use_label_encoder=False,
        eval_metric='auc'
    )

    clf.fit(
        train.drop(['label', 'customer_id'],axis=1), train['label'],
        eval_set=[(train.drop(['label', 'customer_id'],axis=1), train['label'])],
        verbose = 100,
    )

    # eval_metric="auc",
    # callbacks=[lgb.log_evaluation(period=100)],

    # clf.fit(train.drop(['label', 'customer_id'],axis=1), train['label'])

    # with open(os.path.join(model_path, "lgbm_model.pkl"), 'wb') as f:
    with open(os.path.join(model_path, "xgb_model.pkl"), 'wb') as f:
        pickle.dump(clf, f)

def predict_(data_path:str, admission_path:str, model_path:str):
    with open(os.path.join(data_path, "train.pkl"), 'rb') as f:
        train = pickle.load(f)

    with open(os.path.join(data_path, "test.pkl"), 'rb') as f:
        test = pickle.load(f)

    with open(os.path.join(model_path, "lgbm_model.pkl"), 'rb') as f:
        clf = pickle.load(f)

    train = train.reset_index()
    test = test.reset_index()
    all_df=pd.concat([train,test],axis=0)
    train = all_df[all_df['label'].notnull()]
    test = all_df[all_df['label'].isnull()]

    cols = train.columns.tolist()
    cols.remove("label")
    cols.remove('customer_id')

    pred = clf.predict_proba(test.drop(["label", "customer_id"], axis=1))[:, 1] 
    result = pd.read_csv(os.path.join(admission_path, "submission.csv"))
    result['result'] = pred
    result = result.sort_values("result", ascending=False).copy()

    buy_num = 450000
    result.index = range(len(result))
    result.loc[result.index <= buy_num, 'result'] = 1
    result.loc[result.index > buy_num, 'result'] = 0
    result.sort_values("customer_id", ascending=True, inplace=True)
    result.to_csv("./submission.csv", index=False)

if __name__ == "__main__":
    train_("Processed_Data", "model")
    # predict_("Processed_Data", "Data", "model")