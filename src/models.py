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
    model = TumorClassifier(4)
    img = load_image("data/Dataset_BUSI_with_GT/benign/benign (1).png", img_size)
    img = torch.from_numpy(img)
    #datapoint = np.asarray([img, np.array([1, 2, 3, 4])])
    model(img.unsqueeze(0), torch.Tensor([1, 2, 3, 4]).unsqueeze(0))


class BasicImageClassifier(nn.Module):
    """Create Basic Image Classifier for model comparison improvement.

    A class containing a simple classifier for any
    sort of image. The models stemming from this
    class will function to only classify the image
    in one manner alone (malignant or non-malignant).
    This model will not contain any rescaling or
    data augmentation to show how significant the
    accuracy between a model with rescaling and
    data augmentation is against a model without
    any of these.

    """

    def __init__(self):
        """Initialize the image classifier."""
        super(BasicImageClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=(3, 3), stride=2)
        self.mp1 = nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        self.bn1 = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=(3,3), stride=2)
        self.mp2 = nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=(3, 3), stride=2)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=(3, 3), stride=2)
        self.mp3 = nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(2304, 4096)
        self.dropout = nn.Dropout(0.15)
        self.linear2 = nn.Linear(4096, 1000)
        self.linear3 = nn.Linear(1000, 500)
        self.linear4 = nn.Linear(500, 250)
        self.linear5 = nn.Linear(250, 100)
        self.linear6 = nn.Linear(100, 50)
        self.linear7 = nn.Linear(50, 25)
        self.linear8 = nn.Linear(25, 12)
        self.linear9 = nn.Linear(12, 4)

    def forward(self, x):
        """Create Forward Propragration."""
        x = self.conv1(x)
        x = self.mp1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.mp2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.mp3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        x = self.linear5(x)
        x = self.linear6(x)
        x = self.linear7(x)
        x = self.linear8(x)
        x = self.linear9(x)
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



if __name__ == "__main__":
    _main()

#websites:
# https://arxiv.org/pdf/1311.2901.pdf
# https://vitalflux.com/different-types-of-cnn-architectures-explained-examples/
