"""Models Module.

-----------------

This file will contain all of the actualized models
created from the abstract model class(es) made within
the base.py file.
"""
import os
import tracemalloc
import re
from math import floor

# Current issue: Loss is not working properly during training process
import torch
from torch import nn
import torchvision.transforms.functional as TF
import numpy as np
import polars as pl

img_size = (512, 512)
mask_size = (512, 512)
tsize = 8
BATCH_SIZE = 4
validate = False
version=3

def _main():
    model = BasicImageClassifier()
    img = load_image("data/Dataset_BUSI_with_GT/benign/benign (1).png", img_size)
    img = torch.from_numpy(img)
    #datapoint = np.asarray([img, np.array([1, 2, 3, 4])])
    model(img.unsqueeze(0))


class CustomCNN(nn.Module):
    """Custom Convolutional Neural Network.

    A CNN built on pure experimentation, each layer was based on pure
    experimentation with the cancer classification dataset. Although
    this may have started with a simple Convolutional Layer, but will
    evolve based on its accuracy.

    Parameters
    ----------
    n_channels : int
        The number of channels that the image possesses.

    n_classes : int
        The number of classifications within the dataset.
    """

    def __init__(self, n_channels:int=1, n_classes:int=2):
        """Init the Class."""
        super(CustomCNN, self).__init__()
        assert n_classes > 1, "Number of classes must be greater than one."
        # Repeatable Units
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.mp = nn.MaxPool2d(kernel_size=3, stride=2)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout()
        # Convolutions
        self.conv1 = nn.Conv2d(n_channels, 96, kernel_size=7)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3)
        self.conv4 = nn.Conv2d(384, 512, kernel_size=3)
        self.conv5 = nn.Conv2d(512, 640, kernel_size=3)
        self.conv6 = nn.Conv2d(640, 768, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(768, 1024, kernel_size=3, padding=1)
        # Batch Normalizations
        self.bn1 = nn.BatchNorm2d(96)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(384)
        self.bn4 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm2d(640)
        self.bn6 = nn.BatchNorm2d(768)
        self.bn7 = nn.BatchNorm2d(1024)
        # Linears
        self.linear1 = nn.Linear(4096, 2048)
        self.linear2 = nn.Linear(2048, 1024)
        self.linear3 = nn.Linear(1024, 512)
        self.linear4 = nn.Linear(512, 256)
        self.linear5 = nn.Linear(256, 128)
        self.linear6 = nn.Linear(128, 64)
        self.linear7 = nn.Linear(64, n_classes)

    def forward(self, x):
        """Forward pass of the model."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.mp(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.mp(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.mp(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.mp(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.mp(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)
        x = self.mp(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu(x)
        x = self.mp(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear5(x)
        x = self.relu(x)
        x = self.linear6(x)
        x = self.relu(x)
        x = self.linear7(x)
        x = self.softmax(x)
        return x


class AlexNet(nn.Module):
    """Basic Model from 2012 Competition.

    Parameters
    ----------
    n_classes : int
        The number of classes at output.
    n_channels : int
        The number of channels of the image.
    """

    def __init__(self, n_channels:int=1, n_classes:int=2):
        """Init the Class."""
        super(AlexNet, self).__init__()
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.layer1 = nn.Sequential(
                nn.Conv2d(n_channels, 96, kernel_size=11, stride=4, padding=0),
                nn.BatchNorm2d(96),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2)
                )
        self.layer2 = nn.Sequential(
                nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2)
                )
        self.layer3 = nn.Sequential(
                nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(384),
                nn.ReLU()
                )
        self.layer4 = nn.Sequential(
                nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2)
                )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(50_176, 4096),
                nn.ReLU()
                )
        self.fc2 = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU()
                )
        self.fc3 = nn.Sequential(
                nn.Linear(4096, n_classes),
                nn.Softmax(dim=1)
                )

    def forward(self, x):
        """Forward pass of the model."""
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class DoubleConvolution(nn.Module):
    """Does the Double Convolution shown within a unit of the U-Net.

    Parameters
    ----------
    in_channels : Integer

    out_channels : Integer

    """

    def __init__(self, in_channels:int, out_channels:int):
        """Initialize the DC class."""
        super(DoubleConvolution, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass of the model."""
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x


class UNet(nn.Module):
    """Creates a U-Net model for image segmentation.

    Unique class built to develop U-Net models. Inherits from the
    Module class found in pytorch.

    Parameters
    ----------
    in_channels : Integer

    out_channels : Integer

    """

    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        """Initialize the U-Net."""
        super(UNet, self).__init__()
        self.uc = nn.ModuleList()
        self.dc = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate the Down Convolutions
        for feature in features:
            self.dc.append(DoubleConvolution(in_channels, out_channels))
            in_channels = feature

        # Calculate the Up Convolutions
        for feature in reversed(features):
            self.uc.append(nn.ConvTranspose2d(2*feature, feature, kernel_size=2, stride=2))
            self.uc.append(DoubleConvolution(2*feature, feature))

        self.bottleneck = DoubleConvolution(features[-1], 2*features[-1])
        self.final_convolution = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        """Forward pass of u-net."""
        skip_connections = list()
        for down in self.downs:
            x =down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.uc), 2):
            x = self.uc[idx](x)
            skip_connections = skip_connections[idx//2]

            if x.shape != skip_connections.shape:
                x = TF.resize(x, size=skip_connections.shape[2:])

            concat_skip = torch.cat((skip_connections, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_convolution(x)


class TumorClassifier(nn.Module):
    """Tumor Classifier Module that uses both categorical data and image data.

    The machine learning model uses a combination of an image or
    scan in conjunction with categorical data contained within
    the dicom file.

    """

    def __init__(self, cat_input_length:int):
        """Initialize the Module."""
        super(TumorClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2)
        self.mp1 = nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 256, kernel_size=3, stride=2)
        self.mp2 = nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3)
        self.mp3 = nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(43264, 4096)
        self.dropout = nn.Dropout(0.25)
        self.linear2 = nn.Linear(4096, 1000)
        self.linear3 = nn.Linear(1000, 500)
        self.linear4 = nn.Linear(500, 250)
        self.linear5 = nn.Linear(250, 100)
        self.linear6 = nn.Linear(100, 50)
        self.linear7 = nn.Linear(50, 25)
        self.linear8 = nn.Linear(25, 12)
        self.catlinears = nn.ModuleList()
        for i in range(1, floor(np.log2(cat_input_length)) + 1):
            if i - 1 == 0:
                continue
            self.catlinears.append(nn.Linear(int(cat_input_length), int(cat_input_length / 2)))
            cat_input_length = cat_input_length / 2
        self.outlinear = nn.Linear(int(cat_input_length + 12), 2)

    def forward(self, x1, x2):
        """Propagate throughout the machine learning model."""
        x1 = self.conv1(x1)
        x1 = self.mp1(x1)
        x1 = self.bn1(x1)
        x1 = self.conv2(x1)
        x1 = self.mp2(x1)
        x1 = self.bn2(x1)
        x1 = self.conv3(x1)
        x1 = self.conv4(x1)
        x1 = self.mp3(x1)
        x1 = self.flatten(x1)
        x1 = self.linear1(x1)
        x1 = self.dropout(x1)
        x1 = self.linear2(x1)
        x1 = self.linear3(x1)
        x1 = self.linear4(x1)
        x1 = self.linear5(x1)
        x1 = self.linear6(x1)
        x1 = self.linear7(x1)
        x1 = self.linear8(x1)
        for linear in self.catlinears:
            x2 = linear(x2)
        concat = torch.cat((x1, x2), dim=1)
        output = self.outlinear(concat)
        return output


class TutorialNet(nn.Module):
    """Tutorial CNN from Pytorch.

    Parameters
    ----------
    in_channels : int
        The number of channels for input.
    n_classes : int
        The number of classes.
    """

    def __init__(self, in_channels:int=1, n_classes:int=2):
        """Init the Class."""
        super(TutorialNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(250_000, 120) # 7744
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,n_classes)

    def forward(self, x):
        """Forward Pass of the model."""
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    _main()

#websites:
# https://arxiv.org/pdf/1311.2901.pdf
# https://vitalflux.com/different-types-of-cnn-architectures-explained-examples/
