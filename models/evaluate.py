# All metrics in one place


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, fbeta_score

def evaluate_model(y_test, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "f2": fbeta_score(y_test, y_pred, beta=2),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }