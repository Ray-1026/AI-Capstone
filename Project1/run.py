import numpy as np
import torch
import os
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import config as cfg
from models.cnn import Resnet
from models.knn import KNN
from models.vae import VAE, loss_vae
from models.fcn import fcn_autoencoder
from dataset import UnsupervisedDataset, SupervisedDataset, KNNDataset
from utils import same_seeds, unsupervised_score, supervised_score, save_result

same_seeds(42)

model_classes = {
    "cnn": Resnet(),
    "vae": VAE(),
    "fcn": fcn_autoencoder(),
}  # dictionary of models


def train():
    if cfg.model_type == "knn":
        return

    if not os.path.exists("weights"):
        os.mkdir("weights")

    # model
    model = model_classes[cfg.model_type].cuda()

    # dataset & dataloader
    if cfg.model_type == "cnn":
        train_dataset = SupervisedDataset("data/train")
    else:
        train_dataset = UnsupervisedDataset("data/train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=RandomSampler(train_dataset),
    )

    # loss & optimizer
    if cfg.model_type == "cnn":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=cfg.learning_rate)

    best_loss = np.inf

    for epoch in range(cfg.num_epochs):
        model.train()
        train_loss = []

        with tqdm(total=len(train_loader), unit="batch") as tqdm_bar:
            tqdm_bar.set_description(f"Epoch {epoch+1:02d}/{cfg.num_epochs}")
            for data in train_loader:
                if cfg.model_type != "cnn":
                    images = data.float().cuda()

                    if cfg.model_type == "fcn":
                        images = images.view(images.size(0), -1)
                else:
                    images, labels = data
                    images = images.float().cuda()
                    labels = labels.cuda()

                output = model(images)

                # different models use different criterion
                if cfg.model_type == "vae":
                    loss = loss_vae(output[0], images, output[1], output[2], criterion)
                elif cfg.model_type == "fcn":
                    loss = criterion(output, images)
                else:
                    loss = criterion(output, labels)

                train_loss.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tqdm_bar.set_postfix(
                    loss=f"{sum(train_loss)/len(train_loss):.5f}",
                )
                tqdm_bar.update()

        # save best model
        if np.mean(train_loss) < best_loss:
            best_loss = np.mean(train_loss)
            torch.save(model.state_dict(), f"weights/best_{cfg.model_type}.pth")


def test():
    if cfg.model_type == "knn":
        x_train, y_train, x_test, y_test = KNNDataset(
            os.path.join("data", "train"), os.path.join("data", "test")
        )
        model = KNN(x_train, y_train, K=cfg.K)
        y_pred = model.predict(x_test)
        save_result(y_test.cpu().numpy(), y_pred.cpu().numpy())
    else:
        model = model_classes[cfg.model_type].cuda()

        # load weights
        model.load_state_dict(torch.load(f"weights/best_{cfg.model_type}.pth"))

        # dataset & dataloader
        if cfg.model_type == "cnn":
            test_dataset = SupervisedDataset("data/test", train=False)
        else:
            test_dataset = UnsupervisedDataset("data/test", train=False)
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.batch_size,
            sampler=SequentialSampler(test_dataset),
        )

        # loss
        eval_loss = nn.MSELoss(reduction="none")
        model.eval()

        anomality, names, labels = list(), list(), list()
        with torch.no_grad():
            for data, label, name in test_loader:
                img = data.float().cuda()

                if cfg.model_type == "cnn":
                    output = model(img)
                    loss = (output.argmax(1)).int()
                elif cfg.model_type == "fcn":
                    img = img.view(img.size(0), -1)
                    output = model(img, train=False)
                    loss = eval_loss(output, img).sum(1)
                else:
                    output = model(img, train=False)
                    output = output[0]
                    loss = eval_loss(output, img).sum([1, 2, 3])

                anomality.append(loss)
                names.extend([n.split("test\\")[-1] for n in name])
                labels.extend(label)

        # calculate anomaly score
        anomality = torch.cat(anomality, axis=0)
        anomality = torch.sqrt(anomality).reshape(len(test_dataset), 1).cpu().numpy()

        save_result(np.array(labels), anomality.flatten())

    if cfg.model_type != "vae" and cfg.model_type != "fcn":
        supervised_score(cfg.output_filename)
    else:
        unsupervised_score(cfg.output_filename)


if __name__ == "__main__":
    train()
    test()
