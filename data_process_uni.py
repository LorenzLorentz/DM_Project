import os
import argparse
import pandas as pd
import numpy as np
from datetime import date, timedelta
import hydra

def preprocess_agg(data:pd.DataFrame) -> pd.DataFrame:
    feature = pd.DataFrame(
        data.groupby("customer_id")["customer_gender"].last().fillna(0)
    )

    feature[['goods_id_last','goods_status_last','goods_price_last','goods_has_discount_last','goods_list_time_last','goods_delist_time_last']] = data.groupby('customer_id')[['goods_id','goods_status','goods_price','goods_has_discount','goods_list_time','goods_delist_time']].last()
    feature[['order_total_num_last','order_amount_last','order_total_payment_last','order_total_discount_last','order_pay_time_last','order_status_last','order_count_last','is_customer_rate_last','order_detail_status_last', 'order_detail_goods_num_last', 'order_detail_amount_last','order_detail_payment_last', 'order_detail_discount_last']] = data.groupby('customer_id')[['order_total_num', 'order_amount','order_total_payment', 'order_total_discount', 'order_pay_time', 'order_status', 'order_count', 'is_customer_rate','order_detail_status', 'order_detail_goods_num', 'order_detail_amount','order_detail_payment', 'order_detail_discount']].last()
    feature[['member_id_last','member_status_last','is_member_actived_last']] = data.groupby('customer_id')[['member_id','member_status','is_member_actived']].last()
    feature[['goods_price_min','goods_price_max','goods_price_mean','goods_price_std']] = data.groupby('customer_id',as_index = False)['goods_price'].agg({'goods_price_min':'min','goods_price_max':'max','goods_price_mean':'mean','goods_price_std':'std'}).drop(['customer_id'],axis=1)
    feature[['order_total_payment_min','order_total_payment_max','order_total_payment_mean','order_total_payment_std']]= data.groupby('customer_id',as_index = False)['order_total_payment'].agg({'order_total_payment_min':'min','order_total_payment_max':'max','order_total_payment_mean':'mean','order_total_payment_std':'std'}).drop(['customer_id'],axis=1)
    feature[['order_count']] = data.groupby('customer_id',as_index = False)['order_id'].count().drop(['customer_id'],axis=1)
    feature[['goods_count']] = data.groupby('customer_id',as_index = False)['goods_id'].count().drop(['customer_id'],axis=1)
    feature['customer_province'] = data.groupby('customer_id')['customer_province'].last()
    feature['customer_city'] = data.groupby('customer_id')['customer_city'].last()
    feature[['is_customer_rate_mean','is_customer_rate_sum']] = data.groupby('customer_id')['is_customer_rate'].agg([('is_customer_rate_mean',np.mean), ('is_customer_rate_sum',np.sum)])
    feature['discount'] = feature['order_detail_amount_last']/feature['order_detail_payment_last']
    feature[['member_status_mean','member_status_sum']] = data.groupby('customer_id')['member_status'].agg([('member_status_mean',np.mean), ('member_status_sum',np.sum)])
    feature[['order_detail_discount_mean','order_detail_discount_sum']] = data.groupby('customer_id')['order_detail_discount'].agg([('order_detail_discount_mean',np.mean), ('order_detail_discount_sum',np.sum)])      
    feature[['goods_status_mean','goods_status_sum']] = data.groupby('customer_id')['goods_status'].agg([('goods_status_mean',np.mean), ('goods_status_sum',np.sum)])
    feature[['is_member_actived_mean','is_member_actived_sum']] = data.groupby('customer_id')['is_member_actived'].agg([('is_member_actived_mean',np.mean), ('is_member_actived_sum',np.sum)])  
    feature[['order_status_mean','order_status_sum']] = data.groupby('customer_id')['order_status'].agg([('order_status_mean',np.mean), ('order_status_sum',np.sum)])
    feature['order_detail_count'] = data.groupby('customer_id')['customer_id'].count()
    feature[['goods_has_discount_mean','goods_has_discount_sum']] = data.groupby('customer_id')['goods_has_discount'].agg([('goods_has_discount_mean',np.mean), ('goods_has_discount_sum',np.sum)])
    feature[['order_total_payment_mean','order_total_payment_sum']] = data.groupby('customer_id')['order_total_payment'].agg([('order_total_payment_mean',np.mean), ('order_total_payment_sum',np.sum)])
    feature[['order_total_num_mean','order_total_num_sum']] = data.groupby('customer_id')['order_total_num'].agg([('order_total_num_mean',np.mean), ('order_total_num_sum',np.sum)])
    feature['order_pay_time_last'] = pd.to_datetime(feature['order_pay_time_last'])
    feature['order_pay_time_last_m'] = feature['order_pay_time_last'].dt.month
    feature['order_pay_time_last_d'] = feature['order_pay_time_last'].dt.day
    feature['order_pay_time_last_h'] = feature['order_pay_time_last'].dt.hour
    feature['order_pay_time_last_min'] = feature['order_pay_time_last'].dt.minute
    feature['order_pay_time_last_s'] = feature['order_pay_time_last'].dt.second
    feature['order_pay_time_last_weekday'] = feature['order_pay_time_last'].dt.weekday
    t_min=pd.to_datetime('2012-10-11 00:00:00')
    feature['order_pay_time_last_diff'] = (feature['order_pay_time_last'] - t_min).dt.days
    feature['goods_list_time_last'] =pd.to_datetime(feature['goods_list_time_last'])    
    feature['goods_list_time_diff'] = (feature['goods_list_time_last']-t_min).dt.days
    feature['goods_delist_time_last'] =pd.to_datetime(feature['goods_delist_time_last'])    
    feature['goods_delist_time_diff'] = (feature['goods_delist_time_last'] - t_min).dt.days
    feature['goods_time_diff'] =  feature['goods_delist_time_diff'] - feature['goods_list_time_diff']
    feature.drop(['goods_list_time_last','goods_delist_time_last','order_pay_time_last'],axis=1,inplace=True)
    feature.replace([np.inf, -np.inf], np.nan, inplace=True)
    feature.fillna(-999, inplace=True)

    return feature

def get_timespan_features(df: pd.DataFrame, base_date: date, days_to_subtract: int, periods: int, freq: str = 'D') -> pd.DataFrame:
    start_date_dt = base_date - timedelta(days=days_to_subtract)
    all_window_date_cols = pd.date_range(start_date_dt, periods=periods, freq=freq)
    result_df = pd.DataFrame(0, index=df.index, columns=all_window_date_cols)
    cols_present_in_df = [col for col in all_window_date_cols if col in df.columns]
    if cols_present_in_df:
        result_df[cols_present_in_df] = df[cols_present_in_df]
    return result_df

def generate_timeseries_features(payment:pd.DataFrame, goods:pd.DataFrame, target_date:date, is_train:bool=True) -> pd.DataFrame:
    features = {}
    features['customer_id'] = payment.reset_index()['customer_id']

    payment_windows = [14, 30, 60, 91]
    for T in payment_windows:
        current_timespan = get_timespan_features(payment, target_date, T, T)
        decay_coeffs = np.power(0.9, np.arange(T)[::-1])
        features[f'payment_mean_{T}d_decay'] = (current_timespan * decay_coeffs).sum(axis=1).values
        features[f'payment_max_{T}d'] = current_timespan.max(axis=1).values
        features[f'payment_sum_{T}d'] = current_timespan.sum(axis=1).values

        prev_timespan = get_timespan_features(payment, target_date - timedelta(days=7), T, T)
        features[f'payment_mean_{T}d_decay_prev_week'] = (prev_timespan * decay_coeffs).sum(axis=1).values
        features[f'payment_max_{T}d_prev_week'] = prev_timespan.max(axis=1).values
        
        features[f'payment_sales_days_in_last_{T}d'] = (current_timespan != 0).sum(axis=1).values
        features[f'payment_last_sale_days_ago_{T}d'] = T - ((current_timespan!=0)*np.arange(T)).max(axis=1).values
        features[f'payment_first_sale_days_ago_{T}d'] = ((current_timespan!=0)*np.arange(T, 0, -1)).max(axis=1).values
    for i in range(1, 4):
        features[f'payment_sum_month_{i}_before'] = get_timespan_features(payment, target_date, i * 30, 30).sum(axis=1).values

    goods_windows = [21, 49, 84]
    for T in goods_windows:
        current_timespan = get_timespan_features(goods, target_date, T, T)
        features[f'goods_mean_{T}d'] = current_timespan.mean(axis=1).values
        features[f'goods_max_{T}d'] = current_timespan.max(axis=1).values
        features[f'goods_sum_{T}d'] = current_timespan.sum(axis=1).values

        prev_timespan = get_timespan_features(goods, target_date - timedelta(weeks=1), T, T)
        features[f'goods_mean_{T}d_prev_week'] = prev_timespan.mean(axis=1).values
        features[f'goods_max_{T}d_prev_week'] = prev_timespan.max(axis=1).values
        features[f'goods_sum_{T}d_prev_week'] = prev_timespan.sum(axis=1).values
        
        features[f'goods_purchase_days_in_last_{T}d'] = (current_timespan > 0).sum(axis=1).values
        features[f'goods_last_purchase_days_ago_{T}d'] = T - ((current_timespan > 0)*np.arange(T)).max(axis=1).values
        features[f'goods_first_purchase_from_start_{T}d'] = ((current_timespan > 0)*np.arange(T, 0, -1)).max(axis=1).values
    for i in range(1, 4):
        features[f'goods_sum_pseudomonth_{i}_before'] = get_timespan_features(goods, target_date, i * 28, 28).sum(axis=1).values
        
    features_df = pd.DataFrame(features)
    features_df.set_index('customer_id', inplace=True)

    if is_train:
        features_df['label'] = goods[pd.date_range(target_date, periods=30)].max(axis=1).values
        features_df.loc[features_df['label'] > 0, 'label'] = 1

    return features_df

@hydra.main(config_name="train", config_path="config", version_base="1.3")
def preprocess(config):
    data_raw = pd.read_csv(os.path.join(config.data.raw_data_path, "train.csv"))

    train_agg_data = data_raw[data_raw["order_pay_time"] <= "2013-07-31 23:59:59"]
    test_agg_data = data_raw

    label_set = set(data_raw[data_raw["order_pay_time"] > "2013-07-31 23:59:59"]["customer_id"].dropna())

    train_features = pd.DataFrame(index=train_agg_data['customer_id'].unique())
    train_features.index.name = 'customer_id'
    test_features = pd.DataFrame(index=test_agg_data['customer_id'].unique())
    test_features.index.name = 'customer_id'

    if config.data.feature_method == "agg" or config.data.feature_method == "both":
        print("Preprocessing data for aggregated features...")
        train_agg_ft = preprocess_agg(train_agg_data)
        test_agg_ft = preprocess_agg(test_agg_data)
        train_features = train_features.join(train_agg_ft, how='left')
        test_features = test_features.join(test_agg_ft, how='left')

    if config.data.feature_method == "time" or config.data.feature_method == "both":
        print("Preprocessing data for time-series features...")
        ts_data = data_raw[data_raw['order_pay_time'] > '2013-02-01'].copy()
        ts_data['date'] = pd.to_datetime(ts_data['order_pay_time']).dt.date
        
        payment_daily = ts_data.groupby(['date', 'customer_id'], as_index=False)['order_total_payment'].sum()
        payment = payment_daily.set_index(["customer_id", "date"])["order_total_payment"].unstack(level=-1, fill_value=0)
        payment.columns = pd.to_datetime(payment.columns)
        
        goods_daily = ts_data.groupby(['date', 'customer_id'], as_index=False)['order_total_num'].sum()
        goods = goods_daily.set_index(["customer_id", "date"])["order_total_num"].unstack(level=-1, fill_value=0)
        goods.columns = pd.to_datetime(goods.columns)
        
        train_ts_ft = generate_timeseries_features(payment, goods, date(2013, 8, 1), is_train=False) # is_train=False因为标签单独处理

        test_ts_ft = generate_timeseries_features(payment, goods, date(2013, 9, 1), is_train=False)

        train_features = train_features.join(train_ts_ft, how='left')
        test_features = test_features.join(test_ts_ft, how='left')

    train_features['label'] = train_features.index.map(lambda x: int(x in label_set))

    print("Finalizing datasets...")
    train_cols = set(train_features.columns)
    test_cols = set(test_features.columns)
    
    missing_in_test = list(train_cols - test_cols)
    if 'label' in missing_in_test:
        missing_in_test.remove('label')
    for col in missing_in_test:
        if col != 'label':
            test_features[col] = -999

    missing_in_train = list(test_cols - train_cols)
    for col in missing_in_train:
        train_features[col] = -999

    test_features = test_features[train_features.drop('label', axis=1).columns]

    train_features.fillna(-999, inplace=True)
    test_features.fillna(-999, inplace=True)

    train_features.to_pickle(os.path.join(config.data.path, config.data.feature_method+config.data.train_file))
    test_features.to_pickle(os.path.join(config.data.path, config.data.feature_method+config.data.test_file))
    print(f"Processed data saved to {config.data.path}")
    print(f"Train features shape: {train_features.shape}")
    print(f"Test features shape: {test_features.shape}")

if __name__ == "__main__":
    preprocess()