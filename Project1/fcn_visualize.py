import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, SequentialSampler

from models.fcn import fcn_autoencoder
from models.vae import VAE
from dataset import UnsupervisedDataset
from utils import same_seeds

same_seeds(42)


def autoencoder_visual():
    """
    Visualize the reconstructed images from Fully Connected Autoencoder
    """
    model = fcn_autoencoder().cuda()
    model.load_state_dict(torch.load("weights/best_fcn.pth"))
    model.eval()

    # dataset & dataloader
    dataset = UnsupervisedDataset("data/test", train=False)
    dataloader = DataLoader(
        dataset,
        batch_size=60,
        sampler=SequentialSampler(dataset),
    )

    with torch.no_grad():
        for i, (data, label, name) in enumerate(dataloader):
            inputs = data.cuda()
            outputs = model(inputs.view(inputs.size(0), -1), train=False)
            outputs = outputs.view(outputs.size(0), 3, 32, 32)

            # outputs, _, _ = model(inputs, train=False)

            # plot images
            for i in range(60):
                plt.subplot(6, 10, i + 1)
                img1 = inputs[i].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5
                img2 = (outputs[i].permute(1, 2, 0).cpu().numpy() + 1) * 0.5
                img_cat = np.concatenate([img1, img2], axis=1)
                plt.imshow(img_cat)
                plt.axis("off")
            plt.savefig("plot/fcn_visualize.png")
            plt.show()
            break


if __name__ == "__main__":
    autoencoder_visual()
