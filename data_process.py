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

            is_int_convertible = False

            if df[col].isnull().any():
                df[col].fillna(-999, inplace=True)
            
            col_max = df[col].max()
            col_min = df[col].min()

            if not df[col].isnull().all():
                if np.all(np.isfinite(df[col])):
                    if (np.abs(df[col] - df[col].astype(np.int64)).sum()) < 0.01:
                        is_int_convertible = True
                else:
                    is_int_convertible = False
            else:
                 is_int_convertible = True

            if is_int_convertible:
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
    if start_mem_usg > 0:
            pct_reduction = 100 * (start_mem_usg - end_mem_usg) / start_mem_usg
            print(f"Reduced by {pct_reduction:.1f}%")
    
    return df

def get_timespan_features(df:pd.DataFrame, base_date:date, days_to_subtract:int, periods:int, freq:str = 'D') -> pd.DataFrame:
    start_date = base_date - timedelta(days=days_to_subtract)
    date_cols = pd.date_range(start_date, periods=periods, freq=freq)

    existing_cols = [col for col in date_cols if col in df.columns]

    if not existing_cols:
        empty_df_for_agg = pd.DataFrame(0, index=df.index, columns=date_cols)
        return empty_df_for_agg[date_cols]

    return df[existing_cols]

def prepare_dataset(df_payment_pivot:pd.DataFrame, df_goods_pivot:pd.DataFrame, target_date:date, is_train:bool=True) -> pd.DataFrame:
    features = {}
    features['customer_id'] = df_payment_pivot.reset_index()['customer_id']

    print(f"Preparing payment features for target date: {target_date}...")

    payment_time_windows = [14, 30, 60, 91]

    for T in payment_time_windows:
        current_timespan = get_timespan_features(df_payment_pivot, target_date, T, T)
        decay_coeffs = np.power(0.9, np.arange(T)[::-1])

        features[f'payment_mean_{T}d_decay'] = (current_timespan * decay_coeffs).sum(axis=1).values
        features[f'payment_max_{T}d'] = current_timespan.max(axis=1).values
        features[f'payment_sum_{T}d'] = current_timespan.sum(axis=1).values
        features[f'payment_sales_days_in_last_{T}d'] = (current_timespan != 0).sum(axis=1).values

        sales_activity = (current_timespan != 0)
        days_array = np.arange(T) # 0 to T-1

        last_sale_day_index = sales_activity * days_array
        features[f'payment_last_sale_days_ago_{T}d'] = T - last_sale_day_index.where(sales_activity).max(axis=1).fillna(-1).values
        features[f'payment_last_sale_days_ago_{T}d'][features[f'payment_last_sale_days_ago_{T}d'] == T + 1] = T # If max was -1 (no sale)

        first_sale_day_index = sales_activity * np.arange(T, 0, -1) # T for most recent, 1 for oldest
        features[f'payment_first_sale_from_start_{T}d'] = first_sale_day_index.where(sales_activity).max(axis=1).fillna(0).values

        prev_week_target_date = target_date - timedelta(days=7)
        prev_timespan = get_timespan_features(df_payment_pivot, prev_week_target_date, T, T)

        features[f'payment_mean_{T}d_decay_prev_week'] = (prev_timespan * decay_coeffs).sum(axis=1).values
        features[f'payment_max_{T}d_prev_week'] = prev_timespan.max(axis=1).values

    for i in range(1, 4):
        month_timespan = get_timespan_features(df_payment_pivot, target_date, i * 30, 30)
        features[f'payment_sum_month_{i}_before'] = month_timespan.sum(axis=1).values
    
    print(f"Preparing goods features for target date: {target_date}...")
    goods_time_windows = [21, 49, 84]

    for T_goods in goods_time_windows:
        current_goods_timespan = get_timespan_features(df_goods_pivot, target_date, T_goods, T_goods)

        features[f'goods_mean_{T_goods}d'] = current_goods_timespan.mean(axis=1).values
        features[f'goods_max_{T_goods}d'] = current_goods_timespan.max(axis=1).values
        features[f'goods_sum_{T_goods}d'] = current_goods_timespan.sum(axis=1).values
        
        features[f'goods_purchase_days_in_last_{T_goods}d'] = (current_goods_timespan > 0).sum(axis=1).values

        goods_activity = (current_goods_timespan > 0)
        days_array_goods = np.arange(T_goods)
        
        last_purchase_day_index = goods_activity * days_array_goods
        features[f'goods_last_purchase_days_ago_{T_goods}d'] = T_goods - last_purchase_day_index.where(goods_activity).max(axis=1).fillna(-1).values
        features[f'goods_last_purchase_days_ago_{T_goods}d'][features[f'goods_last_purchase_days_ago_{T_goods}d'] == T_goods + 1] = T_goods


        first_purchase_day_index = goods_activity * np.arange(T_goods, 0, -1)
        features[f'goods_first_purchase_from_start_{T_goods}d'] = first_purchase_day_index.where(goods_activity).max(axis=1).fillna(0).values

        prev_week_target_date_goods = target_date - timedelta(weeks=1)
        prev_goods_timespan = get_timespan_features(df_goods_pivot, prev_week_target_date_goods, T_goods, T_goods)

        features[f'goods_mean_{T_goods}d_prev_week'] = prev_goods_timespan.mean(axis=1).values
        features[f'goods_max_{T_goods}d_prev_week'] = prev_goods_timespan.max(axis=1).values
        features[f'goods_sum_{T_goods}d_prev_week'] = prev_goods_timespan.sum(axis=1).values

    for i in range(1, 4):
        goods_month_timespan = get_timespan_features(df_goods_pivot, target_date, i * 28, 28)
        features[f'goods_sum_pseudomonth_{i}_before'] = goods_month_timespan.sum(axis=1).values

    df_features = pd.DataFrame(features)
    df_features.set_index('customer_id', inplace=True)

    if is_train:
        label_timespan = get_timespan_features(df_goods_pivot, target_date + timedelta(days=29), 30, 30, freq='D')
        label_start_date = target_date
        label_date_cols = pd.date_range(label_start_date, periods=30, freq='D')
        existing_label_cols = [col for col in label_date_cols if col in df_goods_pivot.columns]
        
        if not existing_label_cols:
            df_features['label'] = 0
        else:
            label_values = df_goods_pivot[existing_label_cols].max(axis=1).values
            df_features['label'] = np.where(label_values > 0, 1, 0).astype(np.uint8)

    return df_features

def main(data_path:str):
    print("Loading data...")
    train_df = pd.read_csv(os.path.join(data_path, 'train.csv'))

    print("Preprocessing data...")
    train_df = train_df[train_df['order_pay_time'] > '2013-02-01'].copy()
    train_df['date'] = pd.to_datetime(train_df['order_pay_time']).dt.date
    
    df_payment_daily = train_df.groupby(['date', 'customer_id'], as_index=False)['order_total_payment'].sum()
    df_payment_daily.rename(columns={'order_total_payment': 'day_total_payment'}, inplace=True)
    df_payment_pivot = df_payment_daily.set_index(["customer_id", "date"])["day_total_payment"].unstack(level=-1, fill_value=0)
    df_payment_pivot = df_payment_pivot.reindex(sorted(df_payment_pivot.columns), axis=1)
    
    df_goods_daily = train_df.groupby(['date', 'customer_id'], as_index=False)['order_total_num'].sum()
    df_goods_daily.rename(columns={'order_total_num': 'day_total_num'}, inplace=True)
    df_goods_pivot = df_goods_daily.set_index(["customer_id", "date"])["day_total_num"].unstack(level=-1, fill_value=0)
    df_goods_pivot = df_goods_pivot.reindex(sorted(df_goods_pivot.columns), axis=1)

    print("Generating training features...")
    num_time_slots = 4
    first_training_target_date = date(2013, 7, 1)
    list_of_train_feature_dfs = []
    
    for i in range(num_time_slots):
        current_target_date = first_training_target_date + timedelta(days=7 * i)
        print(f"\nGenerating training sample for target date: {current_target_date} (Slot {i+1}/{num_time_slots})")

        X_tmp = prepare_dataset(df_payment_pivot, df_goods_pivot, current_target_date, is_train=True)
        list_of_train_feature_dfs.append(X_tmp)

    X_train_features = pd.concat(list_of_train_feature_dfs, axis=0)

    print("\nReducing memory for X_train_features:")
    X_train_features = reduce_memory_usage(X_train_features)

    print("\nGenerating test features...")
    test_target_date = date(2013, 9, 1)
    print(f"Generating test sample for target date: {test_target_date}")
    X_test_features = prepare_dataset(df_payment_pivot, df_goods_pivot, test_target_date, is_train=False)

    print("\nReducing memory for X_test_features:")
    X_test_features = reduce_memory_usage(X_test_features)

    print("\nSaving processed datasets...")
    X_train_features.to_csv('Processed_Data/X_train_processed.csv')
    X_test_features.to_csv('Processed_Data/X_test_processed.csv')
    print("Processed datasets saved as X_train_processed.csv and X_test_processed.csv")

if __name__ == '__main__':
    data_directory = 'Data/'
    train_file_path = os.path.join(data_directory, 'train.csv')

    if not os.path.exists(train_file_path):
        print(f"'{train_file_path}' not found. Creating dummy data for demonstration.")
        
        num_customers = 100
        num_orders_per_customer = 20
        start_sim_date = datetime(2013, 1, 1)
        
        data = []
        for cust_id in range(1, num_customers + 1):
            for _ in range(num_orders_per_customer):
                days_offset = np.random.randint(0, 300) # orders spread over ~300 days
                order_date = start_sim_date + timedelta(days=int(days_offset))
                payment_time = order_date + timedelta(hours=np.random.randint(0,24))
                
                data.append({
                    'customer_id': cust_id,
                    'order_pay_time': payment_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'order_total_payment': round(np.random.uniform(10, 500), 2),
                    'order_total_num': np.random.randint(1, 10),
                    'order_status': 100 # Assuming a common valid status
                })
        
        dummy_train_df = pd.DataFrame(data)
        dummy_train_df.to_csv(train_file_path, index=False)
        print(f"Dummy 'train.csv' created with {len(dummy_train_df)} records.")
        
    main(data_path=data_directory)
    print("\nProcessing complete.")