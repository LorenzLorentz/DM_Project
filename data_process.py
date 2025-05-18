import os
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime

def reduce_memory_usage(df:pd.DataFrame) -> pd.DataFrame:
    start_mem_usg = df.memory_usage().sum() / 1024**2
    print(f"Memory usage of dataframe is {start_mem_usg:.2f} MB")

    for col in df.columns:
        if df[col].dtype != object and df[col].dtype.name != 'category':
            if df[col].dtype == bool:
                continue

            is_int = False

            if df[col].isnull().any():
                df[col].fillna(-999, inplace=True)

            col_max = df[col].max()
            col_min = df[col].min()

            if not df[col].isnull().all():
                if np.all(np.isfinite(df[col])):
                    if (np.abs(df[col] - df[col].astype(np.int64)).sum()) < 0.01:
                        is_int = True
                else:
                    is_int = False
            else:
                 is_int = True

            if is_int:
                if col_min >= 0:
                    if col_max <= np.iinfo(np.uint8).max:
                        df[col] = df[col].astype(np.uint8)
                    elif col_max <= np.iinfo(np.uint16).max:
                        df[col] = df[col].astype(np.uint16)
                    elif col_max <= np.iinfo(np.uint32).max:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    else:
                        df[col] = df[col].astype(np.int64)
            else:
                df[col] = df[col].astype(np.float32)

    print("___MEMORY USAGE AFTER COMPLETION:___")
    end_mem_usg = df.memory_usage().sum() / 1024**2
    print(f"Memory usage is: {end_mem_usg:.2f} MB")
    pct_reduction = 100 * (start_mem_usg - end_mem_usg) / start_mem_usg
    print(f"Reduced by {pct_reduction:.1f}%")
    
    return df

def get_timespan(df, dt, minus, periods, freq='D'):
    return df[pd.date_range(dt - timedelta(days=minus), periods=periods, freq=freq)]

def get_timespan_features(df: pd.DataFrame, base_date: date, days_to_subtract: int, periods: int, freq: str = 'D') -> pd.DataFrame:
    start_date_dt = base_date - timedelta(days=days_to_subtract)
    all_window_date_cols = pd.date_range(start_date_dt, periods=periods, freq=freq)
    result_df = pd.DataFrame(0, index=df.index, columns=all_window_date_cols)
    cols_present_in_df = [col for col in all_window_date_cols if col in df.columns]

    if cols_present_in_df:
        result_df[cols_present_in_df] = df[cols_present_in_df]
    
    return result_df

def prepare_dataset(payment:pd.DataFrame, goods:pd.DataFrame, target_date:date, is_train:bool=True) -> pd.DataFrame:
    features = {}
    features['customer_id'] = payment.reset_index()['customer_id']

    print(f"Preparing payment features for target date: {target_date}...")

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
    
    print(f"Preparing goods features for target date: {target_date}...")
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

def main(data_path:str, output_dir:str):
    print("Loading data...")
    train_df = pd.read_csv(os.path.join(data_path, 'train.csv'))
    test_df = pd.read_csv(os.path.join(data_path, 'submission.csv'))

    print("Preprocessing data...")
    train_df = train_df[train_df['order_pay_time'] > '2013-02-01'].copy()
    # train_df['date'] = pd.DatetimeIndex(train_df['order_pay_time']).date
    train_df['date'] = pd.to_datetime(train_df['order_pay_time']).dt.date

    payment_daily = train_df.groupby(['date', 'customer_id'], as_index=False)['order_total_payment'].sum()
    payment_daily.rename(columns={'order_total_payment': 'day_total_payment'}, inplace=True)
    payment = payment_daily.set_index(["customer_id", "date"])["day_total_payment"].unstack(level=-1, fill_value=0)
    payment = payment.reindex(sorted(payment.columns), axis=1)
    payment.columns = pd.to_datetime(payment.columns)
    
    goods_daily = train_df.groupby(['date', 'customer_id'], as_index=False)['order_total_num'].sum()
    goods_daily.rename(columns={'order_total_num': 'day_total_num'}, inplace=True)
    goods = goods_daily.set_index(["customer_id", "date"])["day_total_num"].unstack(level=-1, fill_value=0)
    goods = goods.reindex(sorted(goods.columns), axis=1)
    goods.columns = pd.to_datetime(goods.columns)

    print("Generating training features...")
    num_time_slots = 4
    first_training_target_date = date(2013, 7, 1)
    list_of_train_feature_dfs = []
    
    for i in range(num_time_slots):
        current_target_date = first_training_target_date + timedelta(days=7 * i)
        print(f"\nGenerating training sample for target date: {current_target_date} (Slot {i+1}/{num_time_slots})")

        X_tmp = prepare_dataset(payment, goods, current_target_date, is_train=True)
        list_of_train_feature_dfs.append(X_tmp)

    X_train_features = pd.concat(list_of_train_feature_dfs, axis=0)

    print("\nReducing memory for X_train_features:")
    X_train_features = reduce_memory_usage(X_train_features)

    print("\nGenerating test features...")
    test_target_date = date(2013, 9, 1)
    print(f"Generating test sample for target date: {test_target_date}")
    X_test_features = prepare_dataset(payment, goods, test_target_date, is_train=False)

    print("\nReducing memory for X_test_features:")
    X_test_features = reduce_memory_usage(X_test_features)

    print("\nSaving processed datasets...")
    os.makedirs(output_dir, exist_ok=True)
    X_train_features.to_csv(os.path.join(output_dir, 'X_train_processed.csv'))
    X_test_features.to_csv(os.path.join(output_dir, 'X_test_processed.csv'))
    print("Processed datasets saved as X_train_processed.csv and X_test_processed.csv")

if __name__ == '__main__':   
    main(data_path="Data", output_dir="Processed_Data")
    print("\nProcessing complete.")