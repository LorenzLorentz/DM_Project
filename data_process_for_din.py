import pandas as pd
import numpy as np
import torch
import pickle
from pathlib import Path
from collections import deque
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Set, Generator
from tqdm import tqdm

## ----------------- 1. 配置 ----------------- ##

# --- 输入/输出文件路径 ---
DATA_DIR = Path('Data')
PROCESSED_DATA_DIR = Path('Processed_Data')
# 新增一个专门存放分块文件的目录
SUBMISSION_CHUNKS_DIR = PROCESSED_DATA_DIR / 'submission_chunks'
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_CHUNKS_DIR.mkdir(parents=True, exist_ok=True) # 创建分块目录

CSV_FILE_PATH = DATA_DIR / 'train.csv'
SUBMISSION_CSV_PATH = DATA_DIR / 'submission.csv'
TRAIN_DF_PATH = PROCESSED_DATA_DIR / 'train_df.pkl'
TEST_DF_PATH = PROCESSED_DATA_DIR / 'test_df.pkl'
# 注意：最终的submission文件将由多个块组成，这里不再定义单一的输出路径
# SUBMISSION_DF_PATH = PROCESSED_DATA_DIR / 'submission_df.pkl'
ENCODERS_PATH = PROCESSED_DATA_DIR / 'label_encoders.pkl'
VOCAB_SIZES_PATH = PROCESSED_DATA_DIR / 'vocab_sizes.pkl'

# --- 基于时间的分割配置 ---
TRAIN_TARGET_MONTH = '2013-07'
TEST_TARGET_MONTH = '2013-08'
PREDICT_TARGET_MONTH = '2013-09'

# --- 模型超参数 ---
MAX_SEQ_LENGTH = 50
BATCH_SIZE = 128
NEG_SAMPLES_PER_POS = 4
# 新增：用于分批保存的批次大小（每个文件包含的用户数）
SUBMISSION_CHUNK_SIZE = 1000 # 每个分块文件包含1000个用户的预测样本

## ----------------- 2. PyTorch Dataset 和 Collate 函数 (无变动) ----------------- ##
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
    为给定的目标月份创建DIN模型的训练/测试样本。(此函数无变动)
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

# ==================== 主要修改的函数 ====================
def create_and_save_submission_chunks(
    df: pd.DataFrame, 
    target_customers_idx: List[int], 
    all_goods_indices: Set[int], 
    goods_to_class_map: Dict[int, int],
    chunk_size: int,
    output_dir: Path
):
    """
    修改后的函数：为submission用户生成样本，并分块保存为pickle文件，以防止内存爆炸。
    """
    print(f"🚀 开始为目标月份 '{PREDICT_TARGET_MONTH}' 的预测任务生成样本（分块模式）...")
    
    # 1. 准备用户历史数据
    history_df = df[df['month'] < PREDICT_TARGET_MONTH]
    user_history_map = history_df.groupby('customer_id_idx').agg({
        'goods_id_idx': list,
        'goods_class_id_idx': list
    }).to_dict('index')

    all_goods_list = list(all_goods_indices)
    
    chunk_samples = []
    chunk_num = 0
    total_customers = len(target_customers_idx)

    # 2. 遍历目标用户，分块处理
    for i, customer_idx in enumerate(tqdm(target_customers_idx, desc="为所有用户生成并保存预测样本")):
        
        user_data = user_history_map.get(customer_idx, {})
        history_g = list(deque(user_data.get('goods_id_idx', []), maxlen=MAX_SEQ_LENGTH))
        history_c = list(deque(user_data.get('goods_class_id_idx', []), maxlen=MAX_SEQ_LENGTH))
        
        # 3. 为单个用户和所有商品创建样本
        for good_idx in all_goods_list:
            chunk_samples.append({
                'customer_id': customer_idx,
                'history_goods': history_g,
                'history_classes': history_c,
                'candidate_good': good_idx,
                'candidate_class': goods_to_class_map.get(good_idx, 0),
                'label': 0  # 占位符
            })
        
        # 4. 检查是否达到分块大小或是最后一个用户
        # 当处理的用户数达到chunk_size或者这是最后一个用户时，保存当前块
        if (i + 1) % chunk_size == 0 or (i + 1) == total_customers:
            chunk_df = pd.DataFrame(chunk_samples)
            chunk_path = output_dir / f'submission_chunk_{chunk_num}.pkl'
            chunk_df.to_pickle(chunk_path)
            
            print(f"✅ 已保存分块 {chunk_num} 到 '{chunk_path}'. 样本数: {len(chunk_df)}")
            
            # 重置列表和计数器
            chunk_samples = []
            chunk_num += 1
            
    print(f"\n🎉 所有预测样本已分块保存在目录: '{output_dir}'")

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

    # ==================== 修改后的调用逻辑 ====================
    # 6. 生成并分块保存预测用数据
    print("\n🛠️ 正在生成 Submission Set (分块模式)...")
    submission_df_raw = pd.read_csv(SUBMISSION_CSV_PATH)
    known_customers = set(customer_encoder.classes_)
    submission_customers_known = submission_df_raw[submission_df_raw['customer_id'].isin(known_customers)]
    target_customers_idx = customer_encoder.transform(submission_customers_known['customer_id']) + 1
    
    # 清理一下分块目录，防止旧文件干扰
    for old_chunk in SUBMISSION_CHUNKS_DIR.glob('*.pkl'):
        old_chunk.unlink()

    # 调用新的分块保存函数
    create_and_save_submission_chunks(
        df=df,
        target_customers_idx=target_customers_idx,
        all_goods_indices=all_goods,
        goods_to_class_map=goods_to_class_map,
        chunk_size=SUBMISSION_CHUNK_SIZE,
        output_dir=SUBMISSION_CHUNKS_DIR
    )
    # ==========================================================

    print("\n🎉 预处理完成！")

if __name__ == "__main__":
    main()