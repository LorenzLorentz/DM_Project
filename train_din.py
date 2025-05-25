import pandas as pd
import pickle
from torch.utils.data import DataLoader
from pathlib import Path

from model_din import DINModel
from dataset import DinDataset
from data_process_for_din import collate_fn

import hydra

from collections import deque
from tqdm import tqdm

@hydra.main(config_name="train_din", config_path="config", version_base="1.3")
def train(config):
    print("**STEP1 Loading Data")
    train_df = pd.read_pickle(config.data.train_path)
    test_df = pd.read_pickle(config.data.test_path)

    with open(config.data.vocab_path, 'rb') as f:
        vocab_sizes = pickle.load(f)

    print(f"Size of Training set: {len(train_df)}")
    print(f"Size of Test set: {len(test_df)}")
    print(f"Size of Vocab: {vocab_sizes}")

    train_dataset = DinDataset(train_df)
    test_dataset = DinDataset(test_df)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )

    print("**STEP2 Training")

    model = DINModel(
        lr=config.train.lr,
        num_epochs=config.train.num_epochs,
        load_path=None,
        save_path=config.model.save_path,
        num_users=vocab_sizes['n_customers'],
        num_items=vocab_sizes['n_goods'],
        num_categories=vocab_sizes['n_classes'],
        batch_size = config.data.batch_size,
        embedding_dim=config.model.embed_dim,
    )

    model.train(train_loader=train_loader, val_loader=test_loader)

@hydra.main(config_name="train_din", config_path="config", version_base="1.3")
def predict(config):
    print("**STEP 1: Loading Data and Model")
    submission_df = pd.read_pickle(config.data.submission_path)

    with open(config.data.vocab_path, 'rb') as f:
        vocab_sizes = pickle.load(f)
    with open(config.data.encoders_path, 'rb') as f:
        encoders = pickle.load(f)
    customer_encoder = encoders['customer']

    submission_dataset = DinDataset(submission_df)
    submission_loader = DataLoader(
        dataset=submission_dataset,
        batch_size=config.data.batch_size * 4,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )

    model = DINModel(
        lr=config.train.lr,
        num_epochs=config.train.num_epochs,
        load_path=config.model.load_path,
        save_path=None,
        num_users=vocab_sizes['n_customers'],
        num_items=vocab_sizes['n_goods'],
        num_categories=vocab_sizes['n_classes'],
        batch_size = config.data.batch_size,
        embedding_dim=config.model.embed_dim,
    )

    print("**STEP 2: Predicting**")
    predictions = model.predict(data_loader=submission_loader)
    submission_df['probability'] = predictions
    tqdm.pandas(desc="Aggregating predictions")
    customer_predictions = submission_df.groupby('customer_id')['probability'].max().reset_index()
    customer_predictions['result'] = (customer_predictions['probability'] > 0.5).astype(int)
    customer_predictions['customer_id_original'] = customer_encoder.inverse_transform(customer_predictions['customer_id'] - 1)
    submission_template = pd.read_csv(config.data.raw_submission_path)
    final_submission = submission_template[['customer_id']].copy()
    final_submission = final_submission.merge(
        customer_predictions[['customer_id_original', 'result']],
        left_on='customer_id',
        right_on='customer_id_original',
        how='left'
    )
    final_submission.drop(columns=['customer_id_original'], inplace=True)
    final_submission['result'].fillna(0, inplace=True)
    final_submission['result'] = final_submission['result'].astype(int)

    output_path = "submit/submission.csv"
    final_submission.to_csv(output_path, index=False)
    
    print(f"\n🎉 Prediction complete! Submission file saved to '{output_path}'")
    print(final_submission.head())
    print(f"Predicted buys (1): {final_submission['result'].sum()}")

if __name__ == "__main__":
    train()
    # predict()