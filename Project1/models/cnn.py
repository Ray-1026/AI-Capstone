from torch import nn
import torchvision.models as models


class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(512, 2)  # change the last layer to output 2 values

    def forward(self, x):
        x = self.resnet(x)
        return x
