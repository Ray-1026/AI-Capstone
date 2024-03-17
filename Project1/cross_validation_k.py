import numpy as np
import os
import matplotlib.pyplot as plt

import config as cfg
from models.knn import KNN
from dataset import KNNDataset
from utils import same_seeds, supervised_score, save_result
from sklearn.model_selection import KFold

same_seeds(42)


def plot_metrics(accs, pres, recs, f1s, Ks):
    """
    visualize results of cross validation
    """
    plt.figure(figsize=(8, 6))
    met_name = ["Accuracy", "Precision", "Recall", "F1"]
    for i, met in enumerate([accs, pres, recs, f1s]):
        plt.subplot(2, 2, i + 1)
        ks, means, stds = [], [], []
        for k, m in met.items():
            plt.scatter([k] * len(m), m, color="lightskyblue")
            ks.append(k)
            means.append(np.mean(m))
            stds.append(np.std(m))
        plt.errorbar(ks, means, yerr=stds, color="orange")
        plt.title(f"{met_name[i]}")
    plt.savefig("plot/knn_cross_validation.png")
    plt.show()


def knn_cross_validation():
    """
    find the best K for KNN model using cross validation
    """
    Ks = range(1, 11)
    accs, pres, recs, f1s = (
        {k: [] for k in Ks},
        {k: [] for k in Ks},
        {k: [] for k in Ks},
        {k: [] for k in Ks},
    )
    x_train, y_train, x_test, y_test = KNNDataset(
        os.path.join("data", "train"), os.path.join("data", "test")
    )

    kf = KFold(n_splits=5, shuffle=True)
    for train_index, test_index in kf.split(x_train):
        # split train and validation set
        x_train_fold, x_val_fold = x_train[train_index], x_train[test_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[test_index]

        for K in Ks:
            model = KNN(x_train_fold, y_train_fold, K=K)
            y_pred = model.predict(x_val_fold)
            save_result(y_val_fold.cpu().numpy(), y_pred.cpu().numpy())
            ret = supervised_score(cfg.output_filename, print_result=False)
            accs[K].append(ret[0])
            pres[K].append(ret[1])
            recs[K].append(ret[2])
            f1s[K].append(ret[3])

    plot_metrics(accs, pres, recs, f1s, Ks)

    # find the best K
    best_K = 0
    best_acc = 0
    for K in Ks:
        acc = np.mean(accs[K])
        prec = np.mean(pres[K])
        rec = np.mean(recs[K])
        f1 = np.mean(f1s[K])
        print(
            f"K: {K:2d}, Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}"
        )
        if acc > best_acc:
            best_acc = acc
            best_K = K

    print(f"best K: {best_K}", end=", ")

    # test the best K
    model = KNN(x_train, y_train, K=best_K)
    y_pred = model.predict(x_test)
    save_result(y_test.cpu().numpy(), y_pred.cpu().numpy())
    supervised_score(cfg.output_filename, print_result=True)


if __name__ == "__main__":
    knn_cross_validation()
