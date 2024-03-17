import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import config as cfg
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
)
from sklearn.preprocessing import MinMaxScaler


def same_seeds(myseed):
    """
    Set the random seeds to make the experiments reproducible.
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)


def save_result(y_test, y_pred):
    """
    Save the results to a csv file.
    """
    result = pd.DataFrame(
        {
            "label": y_test,
            "score": y_pred,
        }
    )
    result = result.sort_values(by="score", ascending=False)
    result.to_csv(cfg.output_filename, index=False)


def unsupervised_score(file_name):
    """
    Calculate the unsupervised evaluation metrics.
    """
    csv_file = file_name
    df = pd.read_csv(csv_file)
    df["label"] = df["label"].astype(int)
    df["score"] = MinMaxScaler().fit_transform(df[["score"]])

    y_true = df["label"]
    y_pred = df["score"]

    roc_auc = roc_auc_score(y_true, y_pred)
    print(f"roc_auc: {roc_auc:.4f}")

    # draw ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.figure()
    lw = 2
    plt.plot(
        fpr, tpr, color="darkorange", lw=lw, label="ROC curve (area = %0.4f)" % roc_auc
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.legend(loc="lower right")
    plt.savefig(f"plot/{cfg.model_type}_roc.png")
    plt.show()


def supervised_score(file_name, print_result=True):
    """
    Calculate the supervised evaluation metrics.
    """
    csv_file = file_name
    df = pd.read_csv(csv_file)
    y_true = df["label"]
    y_pred = df["score"]

    # precision, recall, F1-score
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    if print_result:
        print(
            f"acc: {np.mean((y_true == y_pred)):.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}"
        )
    else:
        return np.mean((y_true == y_pred)), precision, recall, f1
