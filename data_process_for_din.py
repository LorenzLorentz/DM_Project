import pandas as pd
import numpy as np
import torch
import pickle
from pathlib import Path
from collections import deque
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Set
import hydra
from dataset import DinDataset
from tqdm import tqdm

## ----------------- 1. 配置 ----------------- ##

# --- 输入/输出文件路径 ---
DATA_DIR = Path('Data')
PROCESSED_DATA_DIR = Path('Processed_Data')
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILE_PATH = DATA_DIR / 'train.csv'
SUBMISSION_CSV_PATH = DATA_DIR / 'submission.csv' # 新增：submission文件路径
TRAIN_DF_PATH = PROCESSED_DATA_DIR / 'train_df.pkl'
TEST_DF_PATH = PROCESSED_DATA_DIR / 'test_df.pkl'
SUBMISSION_DF_PATH = PROCESSED_DATA_DIR / 'submission_df.pkl' # 新增：为submission生成的数据
ENCODERS_PATH = PROCESSED_DATA_DIR / 'label_encoders.pkl'
VOCAB_SIZES_PATH = PROCESSED_DATA_DIR / 'vocab_sizes.pkl'

# --- 基于时间的分割配置 ---
TRAIN_TARGET_MONTH = '2013-07'
TEST_TARGET_MONTH = '2013-08'
PREDICT_TARGET_MONTH = '2013-09' # 新增：预测的目标月份

# --- 模型超参数 ---
MAX_SEQ_LENGTH = 50
BATCH_SIZE = 128
NEG_SAMPLES_PER_POS = 4

## ----------------- 2. PyTorch Dataset 和 Collate 函数 ----------------- ##

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    自定义的collate函数，用于处理变长序列。
    """
    batch_dict = {key: [d[key] for d in batch] for key in batch[0]}
    batch_dict['history_goods'] = pad_sequence(batch_dict['history_goods'], batch_first=True, padding_value=0)
    batch_dict['history_classes'] = pad_sequence(batch_dict['history_classes'], batch_first=True, padding_value=0)

    for key in batch_dict:
        if key not in ['history_goods', 'history_classes']:
            batch_dict[key] = torch.stack(batch_dict[key], dim=0)
    return batch_dict


## ----------------- 3. 数据处理核心函数 ----------------- ##

def create_din_samples(df: pd.DataFrame, target_month: str, all_goods_indices: Set[int], goods_to_class_map: Dict[int, int]) -> pd.DataFrame:
    """
    为给定的目标月份创建DIN模型的训练/测试样本。
    """
    print(f"正在为目标月份 '{target_month}' 生成样本...")
    history_df = df[df['month'] < target_month]
    target_df = df[df['month'] == target_month]

    user_history = history_df.groupby('customer_id_idx').agg({
        'goods_id_idx': list,
        'goods_class_id_idx': list
    }).reset_index()
    user_history.rename(columns={'goods_id_idx': 'history_goods', 'goods_class_id_idx': 'history_classes'}, inplace=True)

    positive_samples = target_df.groupby('customer_id_idx')['goods_id_idx'].apply(list).reset_index(name='target_goods')
    merged_df = pd.merge(positive_samples, user_history, on='customer_id_idx', how='left')
    merged_df['history_goods'] = merged_df['history_goods'].apply(lambda x: x if isinstance(x, list) else [])
    merged_df['history_classes'] = merged_df['history_classes'].apply(lambda x: x if isinstance(x, list) else [])

    din_samples = []
    for _, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="Generating train/test samples"):
        history_g = list(deque(row['history_goods'], maxlen=MAX_SEQ_LENGTH))
        history_c = list(deque(row['history_classes'], maxlen=MAX_SEQ_LENGTH))
        
        for good_idx in row['target_goods']:
            din_samples.append({
                'customer_id': row['customer_id_idx'],
                'history_goods': history_g,
                'history_classes': history_c,
                'candidate_good': good_idx,
                'candidate_class': goods_to_class_map.get(good_idx, 0),
                'label': 1
            })

        purchased_set = set(row['target_goods'])
        negative_pool = list(all_goods_indices - purchased_set)
        num_neg_to_sample = min(len(negative_pool), len(purchased_set) * NEG_SAMPLES_PER_POS)
        if num_neg_to_sample > 0:
            neg_good_indices = np.random.choice(negative_pool, size=num_neg_to_sample, replace=False)
            for neg_good_idx in neg_good_indices:
                din_samples.append({
                    'customer_id': row['customer_id_idx'],
                    'history_goods': history_g,
                    'history_classes': history_c,
                    'candidate_good': neg_good_idx,
                    'candidate_class': goods_to_class_map.get(neg_good_idx, 0),
                    'label': 0
                })
    return pd.DataFrame(din_samples)

def create_submission_samples(df: pd.DataFrame, target_customers_idx: List[int], all_goods_indices: Set[int], goods_to_class_map: Dict[int, int]) -> pd.DataFrame:
    """
    为 submission.csv 中的用户生成预测所需的样本。
    """
    print(f"正在为目标月份 '{PREDICT_TARGET_MONTH}' 的预测任务生成样本...")

    # 1. 历史数据为预测月之前的所有数据
    history_df = df[df['month'] < PREDICT_TARGET_MONTH]
    
    # 2. 按用户聚合其全部历史购买记录
    user_history = history_df.groupby('customer_id_idx').agg({
        'goods_id_idx': list,
        'goods_class_id_idx': list
    }).reset_index()
    user_history.rename(columns={'goods_id_idx': 'history_goods', 'goods_class_id_idx': 'history_classes'}, inplace=True)

    # 3. 筛选出 submission.csv 中指定的用户
    target_user_history = user_history[user_history['customer_id_idx'].isin(target_customers_idx)]

    # 4. 为每个目标用户和每个可能的商品生成一条记录
    submission_samples = []
    all_goods_list = list(all_goods_indices)
    
    for _, row in tqdm(target_user_history.iterrows(), total=len(target_user_history), desc="Generating submission samples"):
        history_g = list(deque(row['history_goods'], maxlen=MAX_SEQ_LENGTH))
        history_c = list(deque(row['history_classes'], maxlen=MAX_SEQ_LENGTH))
        
        # 为该用户和所有商品创建候选样本
        for good_idx in all_goods_list:
            submission_samples.append({
                'customer_id': row['customer_id_idx'],
                'history_goods': history_g,
                'history_classes': history_c,
                'candidate_good': good_idx,
                'candidate_class': goods_to_class_map.get(good_idx, 0),
                'label': 0 # label 在预测时无用，仅作占位
            })

    return pd.DataFrame(submission_samples)


## ----------------- 4. 主执行逻辑 ----------------- ##
def main():
    """主预处理流程"""
    print("🚀 开始DIN数据预处理...")

    # 1. 加载和初步清洗数据
    df = pd.read_csv(CSV_FILE_PATH)
    df['order_pay_time'] = pd.to_datetime(df['order_pay_time'], errors='coerce')
    df.dropna(subset=['customer_id', 'goods_id', 'goods_class_id', 'order_pay_time'], inplace=True)
    df['month'] = df['order_pay_time'].dt.strftime('%Y-%m')
    df.sort_values(by='order_pay_time', inplace=True)
    df.reset_index(drop=True, inplace=True)
    print("✅ 数据加载、清洗并按时间排序完成。")

    # 2. 特征编码
    customer_encoder = LabelEncoder()
    goods_encoder = LabelEncoder()
    class_encoder = LabelEncoder()

    df['customer_id_idx'] = customer_encoder.fit_transform(df['customer_id']) + 1
    df['goods_id_idx'] = goods_encoder.fit_transform(df['goods_id']) + 1
    df['goods_class_id_idx'] = class_encoder.fit_transform(df['goods_class_id']) + 1
    print("✅ 用户、商品和类别ID编码完成。")

    # 3. 保存编码器和词汇表大小
    encoders = {
        'customer': customer_encoder, 'goods': goods_encoder, 'class': class_encoder
    }
    with open(ENCODERS_PATH, 'wb') as f: pickle.dump(encoders, f)
    print(f"✅ 编码器已保存到: '{ENCODERS_PATH}'")
    
    vocab_sizes = {
        'n_customers': len(customer_encoder.classes_) + 1,
        'n_goods': len(goods_encoder.classes_) + 1,
        'n_classes': len(class_encoder.classes_) + 1
    }
    with open(VOCAB_SIZES_PATH, 'wb') as f: pickle.dump(vocab_sizes, f)
    print(f"词汇表大小: {vocab_sizes}")
    
    # 4. 准备生成样本所需的数据结构
    goods_to_class_map = dict(zip(df['goods_id_idx'], df['goods_class_id_idx']))
    all_goods = set(df['goods_id_idx'].unique())
    
    # 5. 生成并保存训练和测试数据
    print("\n🛠️ 正在生成训练集...")
    train_df = create_din_samples(df, TRAIN_TARGET_MONTH, all_goods, goods_to_class_map)
    train_df.to_pickle(TRAIN_DF_PATH)
    print(f"✅ 训练集已保存到 '{TRAIN_DF_PATH}'. 样本数: {len(train_df)}")

    print("\n🛠️ 正在生成测试集...")
    test_df = create_din_samples(df, TEST_TARGET_MONTH, all_goods, goods_to_class_map)
    test_df.to_pickle(TEST_DF_PATH)
    print(f"✅ 测试集已保存到 '{TEST_DF_PATH}'. 样本数: {len(test_df)}")

    # 6. 新增：生成并保存预测用数据
    print("\n🛠️ 正在生成 Submission Set...")
    submission_df_raw = pd.read_csv(SUBMISSION_CSV_PATH)
    # 将 submission.csv 中的 customer_id 转换为我们训练时使用的 index
    # 注意：只转换那些在训练数据中出现过的 customer_id
    known_customers = set(customer_encoder.classes_)
    submission_customers_known = submission_df_raw[submission_df_raw['customer_id'].isin(known_customers)]
    target_customers_idx = customer_encoder.transform(submission_customers_known['customer_id']) + 1
    
    submission_df = create_submission_samples(df, target_customers_idx, all_goods, goods_to_class_map)
    submission_df.to_pickle(SUBMISSION_DF_PATH)
    print(f"✅ Submission set 已保存到 '{SUBMISSION_DF_PATH}'. 样本数: {len(submission_df)}")

    print("\n🎉 预处理完成！")

if __name__ == "__main__":
    main()