# draw t-SNE
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler
from sklearn.manifold import TSNE

import config as cfg
from models.cnn import Resnet
from dataset import SupervisedDataset
from utils import same_seeds

same_seeds(42)


def tSNE_visualize():
    """
    Visualize the t-SNE of the features from each layer
    """
    model = Resnet().cuda()
    model.load_state_dict(torch.load(f"weights/best_cnn.pth"))
    model.eval()

    print(model)  # print model architecture

    test_dataset = SupervisedDataset("data/test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        sampler=SequentialSampler(test_dataset),
    )

    def hook(module, input, output):
        """
        hook for extracting features
        """
        temp.append(output.clone().detach())

    handles = []
    handles.append(model.resnet.layer1.register_forward_hook(hook))
    handles.append(model.resnet.layer2.register_forward_hook(hook))
    handles.append(model.resnet.layer3.register_forward_hook(hook))
    handles.append(model.resnet.layer4.register_forward_hook(hook))
    handles.append(model.resnet.fc.register_forward_hook(hook))

    features_tsnes = []

    # input
    features = []
    labels = []
    for img, label in tqdm(test_loader):
        with torch.no_grad():
            logits = img.view(img.size()[0], -1)
            labels.extend(label.numpy())
            logits = np.squeeze(logits.cpu().numpy())
            features.extend(logits)
    features = np.array(features)
    features_tsnes.append(TSNE(n_components=2).fit_transform(features))

    # layers 1-4 and output
    for handle in handles:
        features = []
        labels = []
        with torch.no_grad():
            for img, label in tqdm(test_loader):
                temp = []
                _ = model(img.cuda())
                logits = temp[0].view(temp[0].size()[0], -1)
                labels.extend(label.numpy())
                logits = np.squeeze(logits.cpu().numpy())
                features.extend(logits)
        handle.remove()

        features = np.array(features)
        features_tsnes.append(TSNE(n_components=2).fit_transform(features))

    # draw t-SNE
    plt.figure(figsize=(10, 6))
    for i in range(len(features_tsnes)):
        plt.subplot(2, 3, i + 1)
        for label in np.unique(labels):
            plt.scatter(
                features_tsnes[i][labels == label, 0],
                features_tsnes[i][labels == label, 1],
                label=label,
                s=10,
            )
        if i == 0:
            plt.title("Input")
        elif i == 5:
            plt.title("Output")
        else:
            plt.title(f"Layer {i}")
        plt.legend()
    plt.savefig("plot/tsne.png")
    plt.show()


if __name__ == "__main__":
    tSNE_visualize()
