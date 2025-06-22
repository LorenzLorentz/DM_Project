import re
import matplotlib.pyplot as plt
from typing import List, Dict

def parse_lgb_log(file_path: str) -> List[float]:
    auc_scores = []

    with open(file_path, 'r') as f:
        for line in f:
            match = re.search(r"valid_0's auc: ([\d\.]+)", line)
            if match:
                auc_scores.append(float(match.group(1)))

    return auc_scores

def parse_xgb_log(file_path:str) -> List[float]:
    auc_scores = []

    with open(file_path, 'r') as f:
        for line in f:
            match = re.search(r"validation_0-auc:([\d\.]+)", line)
            if match:
                auc_scores.append(float(match.group(1)))

    return auc_scores

def parse_mlp_log(file_path:str) -> List[float]:
    auc_scores = []

    with open(file_path, 'r') as f:
        for line in f:
            match = re.search(r"auc = ([\d\.]+)", line)
            if match:
                auc_scores.append(float(match.group(1)))

    return auc_scores

def parse_reg_log(file_path:str) -> List[float]:
    auc_scores = []

    with open(file_path, 'r') as f:
        for line in f:
            match = re.search(r"auc = ([\d\.]+)", line)
            if match:
                auc_scores.append(float(match.group(1)))

    return auc_scores

def plot_auc_comparison(data: Dict[str, List[float]], output_filename:str = 'auc.png', algorithm:str=None):
    fig, ax = plt.subplots(figsize=(12, 8))

    for label, scores in data.items():
        if scores:
            ax.plot(range(1, len(scores) + 1), scores, label=label.capitalize())
    
    ax.set_title(f'Comparison of AUC Scores for Different Data Preprocessing Methods for algorithm {algorithm}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('AUC Score')
    ax.set_ylim(0, 1.0)

    ax.legend(title='Preprocessing Method', fontsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig("figs/"+algorithm+output_filename, dpi=300)

if __name__ == '__main__':
    lgb_logs = {
        'agg': 'lgb_agg.log',
        'time': 'lgb_time.log',
        'both': 'lgb_both.log'
    }

    xgb_logs = {
        'agg': 'xgb_agg.log',
        'time': 'xgb_time.log',
        'both': 'xgb_both.log'
    }

    mlp_logs = {
        'agg': 'mlp_agg.log',
        'time': 'mlp_time.log',
        'both': 'mlp_both.log'
    }

    reg_logs = {
        'agg': 'reg_agg.log',
        'time': 'reg_time.log',
        'both': 'reg_both.log'
    }

    """
    lgb_auc_data = {}
    for label, filepath in lgb_logs.items():
        lgb_auc_data[label] = parse_lgb_log(filepath)    
    plot_auc_comparison(lgb_auc_data, algorithm="lgb")
    """
    
    """
    xgb_auc_data = {}
    for label, filepath in xgb_logs.items():
        xgb_auc_data[label] = parse_xgb_log(filepath)    
    plot_auc_comparison(xgb_auc_data, algorithm="xgb")
    """

    # """
    mlp_auc_data = {}
    for label, filepath in mlp_logs.items():
        mlp_auc_data[label] = parse_mlp_log(filepath)
    plot_auc_comparison(mlp_auc_data, algorithm="mlp")
    # """

    reg_auc_data = {}
    for label, filepath in reg_logs.items():
        reg_auc_data[label] = parse_mlp_log(filepath)
    plot_auc_comparison(reg_auc_data, algorithm="reg")