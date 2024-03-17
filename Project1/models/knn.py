import torch


class KNN:
    def __init__(self, x_train, y_train, K=1):
        self.x_train = x_train
        self.y_train = y_train
        self.K = K

    def predict(self, x_test):
        dists = self.compute_distance(self.x_train, x_test)
        return self.predict_labels(dists, self.y_train)

    def compute_distance(self, x_train, x_test):
        """
        Compute squared Euclidean distance using vectorization
        """
        num_train = x_train.shape[0]
        num_test = x_test.shape[0]
        x_train_flat = x_train.view(num_train, -1)
        x_test_flat = x_test.view(num_test, -1)

        dists = x_train.new_zeros(num_test, num_train)
        dists = (
            torch.sum(x_train_flat**2, dim=1, keepdim=True)
            + torch.sum(x_test_flat**2, dim=1)
            - 2 * torch.mm(x_train_flat, x_test_flat.t())
        )
        return dists

    def predict_labels(self, dists, y_train):
        """
        Find the k nearest neighbors and predict the labels
        """
        _, num_test = dists.shape
        y_pred = torch.zeros(num_test, dtype=torch.int64)

        for i in range(num_test):
            top_k = torch.topk(dists[:, i], self.K, largest=False)
            top_k_labels = y_train[top_k.indices]
            y_pred[i] = torch.bincount(top_k_labels).argmax()

        return y_pred
