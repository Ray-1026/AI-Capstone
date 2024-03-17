import torch
import os
from sklearn.decomposition import PCA

import config as cfg
from models.knn import KNN
from dataset import KNNDataset
from utils import same_seeds, supervised_score, save_result


same_seeds(42)


def KNN_PCA():
    """
    PCA + KNN
    """
    x_train, y_train, x_test, y_test = KNNDataset(
        os.path.join("data", "train"), os.path.join("data", "test")
    )

    # use PCA to reduce dimension to 2
    pca = PCA(n_components=2)
    x_train_pca = pca.fit_transform(x_train.view(x_train.size(0), -1))
    x_test_pca = pca.fit_transform(x_test.view(x_test.size(0), -1))
    x_train_pca = torch.tensor(x_train_pca)
    x_test_pca = torch.tensor(x_test_pca)

    model = KNN(x_train_pca, y_train, K=cfg.K)
    y_pred = model.predict(x_test_pca)
    save_result(y_test.cpu().numpy(), y_pred.cpu().numpy())
    supervised_score(cfg.output_filename)


if __name__ == "__main__":
    KNN_PCA()
