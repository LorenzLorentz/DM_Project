from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

def print_metrics(y_true, y_pred_proba, y_pred_binary, phase="Validation"):
    auc = roc_auc_score(y_true, y_pred_proba)
    acc = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    print(f"{phase} Metrics: AUC={auc:.4f} | Acc={acc:.4f} | Precision={precision:.4f} | Recall={recall:.4f} | F1={f1:.4f}")
    return {"auc": auc, "accuracy": acc, "precision": precision, "recall": recall, "f1": f1}