import numpy as np
import torch
import os
import torchvision.models as models
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import config as cfg
from dataset import SupervisedDataset
from utils import same_seeds, supervised_score, save_result


same_seeds(42)


class Resnet_M(nn.Module):
    def __init__(self):
        super(Resnet_M, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(512, 1)  # change the last layer to output 1 value

    def forward(self, x):
        x = self.resnet(x)
        return x


def train():
    if not os.path.exists("weights"):
        os.mkdir("weights")

    model = Resnet_M().cuda()

    train_dataset = SupervisedDataset("data/train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=RandomSampler(train_dataset),
    )

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=cfg.learning_rate)

    best_loss = np.inf

    for epoch in range(cfg.num_epochs):
        model.train()
        train_loss = []

        with tqdm(total=len(train_loader), unit="batch") as tqdm_bar:
            tqdm_bar.set_description(f"Epoch {epoch+1:02d}/{cfg.num_epochs}")
            for data in train_loader:
                images, labels = data
                images = images.float().cuda()
                labels = labels.cuda().view(-1, 1).float()

                output = model(images)

                loss = criterion(output, labels)
                train_loss.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tqdm_bar.set_postfix(
                    loss=f"{sum(train_loss)/len(train_loss):.5f}",
                )
                tqdm_bar.update()

        if np.mean(train_loss) < best_loss:
            best_loss = np.mean(train_loss)
            torch.save(model.state_dict(), f"weights/best_resnet_m.pth")


def test():
    model = Resnet_M().cuda()
    model.load_state_dict(torch.load(f"weights/best_resnet_m.pth"))
    model.eval()

    test_dataset = SupervisedDataset("data/test", train=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        sampler=SequentialSampler(test_dataset),
    )

    threshold = 0.5  # threshold for evaluating anomality

    anomality, names, labels = list(), list(), list()
    with torch.no_grad():
        for data, label, name in test_loader:
            img = data.float().cuda()

            output = model(img)
            loss = output > threshold

            anomality.append(loss)
            names.extend([n.split("test\\")[-1] for n in name])
            labels.extend(label)

    anomality = torch.cat(anomality, axis=0)
    anomality = torch.sqrt(anomality).reshape(len(test_dataset), 1).cpu().numpy()

    save_result(np.array(labels), anomality.flatten())
    supervised_score(cfg.output_filename)


if __name__ == "__main__":
    train()
    test()
