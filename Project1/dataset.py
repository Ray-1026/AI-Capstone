import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os


class UnsupervisedDataset(Dataset):
    """
    Custom dataset for unsupervised learning (VAE, Fully Connected Autoencoder)
    """

    def __init__(self, root, train=True):
        self.root = root
        self.train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.to(torch.float32)),
                transforms.Lambda(lambda x: 2.0 * x - 1.0),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(45),
                transforms.RandomCrop(32, padding=4),
            ]
        )
        self.test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.to(torch.float32)),
                transforms.Lambda(lambda x: 2.0 * x - 1.0),
            ]
        )

        self.train = train

        if self.train:
            normal = os.listdir(os.path.join(self.root, "normal"))
            self.images = [os.path.join(self.root, "normal", img) for img in normal]
        else:
            normal = os.listdir(os.path.join(self.root, "normal"))
            abnormal = os.listdir(os.path.join(self.root, "abnormal"))
            self.images = [os.path.join(self.root, "normal", img) for img in normal] + [
                os.path.join(self.root, "abnormal", img) for img in abnormal
            ]
            self.labels = [0] * len(normal) + [1] * len(abnormal)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        if self.train:
            image = self.train_transform(image)
        else:
            image = self.test_transform(image)

        if self.train:
            return image
        else:
            return image, self.labels[idx], self.images[idx]


class SupervisedDataset(Dataset):
    """
    Custom dataset for supervised learning (ResNet)
    """

    def __init__(self, root, train=True):
        self.root = root
        self.train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.to(torch.float32)),
                transforms.Lambda(lambda x: 2.0 * x - 1.0),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(45),
                # transforms.RandomCrop(32, padding=4),
            ]
        )
        self.test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.to(torch.float32)),
                transforms.Lambda(lambda x: 2.0 * x - 1.0),
            ]
        )

        self.train = train

        normal = os.listdir(os.path.join(self.root, "normal"))
        abnormal = os.listdir(os.path.join(self.root, "abnormal"))
        self.images = [os.path.join(self.root, "normal", img) for img in normal] + [
            os.path.join(self.root, "abnormal", img) for img in abnormal
        ]
        self.labels = [0] * len(normal) + [1] * len(abnormal)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        if self.train:
            image = self.train_transform(image)
        else:
            image = self.test_transform(image)

        if self.train:
            return image, self.labels[idx]
        else:
            return image, self.labels[idx], self.images[idx]


def KNNDataset(train_path, test_path, aug=False):
    """
    Custom dataset for KNN model
    """
    x_train = [
        os.path.join(train_path, "normal", i)
        for i in os.listdir(os.path.join(train_path, "normal"))
    ] + [
        os.path.join(train_path, "abnormal", i)
        for i in os.listdir(os.path.join(train_path, "abnormal"))
    ]
    y_train = [0] * len(os.listdir(os.path.join(train_path, "normal"))) + [1] * len(
        os.listdir(os.path.join(train_path, "abnormal"))
    )

    x_test = [
        os.path.join(test_path, "normal", i)
        for i in os.listdir(os.path.join(test_path, "normal"))
    ] + [
        os.path.join(test_path, "abnormal", i)
        for i in os.listdir(os.path.join(test_path, "abnormal"))
    ]
    y_test = [0] * len(os.listdir(os.path.join(test_path, "normal"))) + [1] * len(
        os.listdir(os.path.join(test_path, "abnormal"))
    )

    if aug:
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(45),
                transforms.RandomCrop(32, padding=4),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    x_train = [Image.open(i) for i in x_train]
    x_train = torch.stack([train_transform(i) for i in x_train])
    y_train = torch.tensor(y_train)
    x_test = [Image.open(i) for i in x_test]
    x_test = torch.stack([test_transform(i) for i in x_test])
    y_test = torch.tensor(y_test)

    return x_train, y_train, x_test, y_test
