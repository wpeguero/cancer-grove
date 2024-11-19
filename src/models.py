"""Models Module.

-----------------

This file will contain all of the actualized models
created from the abstract model class(es) made within
the base.py file.
"""

from math import floor

import numpy as np

# Current issue: Loss is not working properly during training process
import torch
import torchvision.transforms.functional as TF
from torch import nn

from utils import load_image

img_size = (512, 512)
mask_size = (512, 512)
tsize = 8
BATCH_SIZE = 4
validate = False
version = 3


def _main():
    model = nn.Sequential(
        InceptionStem(3),
        InceptionA(384),
        ReductionA(384),
        InceptionB(1024),
        ReductionB(1024),
        InceptionC(1536),
        nn.Flatten(),
        nn.AvgPool2d(kernel_size=3, stride=2),
        nn.Dropout(0.8),
        nn.Softmax(4),
    )
    exit()
    img = load_image("data/Dataset_BUSI_with_GT/benign/benign (1).png", img_size)
    img = torch.from_numpy(img)
    # datapoint = np.asarray([img, np.array([1, 2, 3, 4])])
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

    def __init__(self, n_channels: int = 1, n_classes: int = 2):
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

    def __init__(self, n_channels: int = 1, n_classes: int = 2):
        """Init the Class."""
        super(AlexNet, self).__init__()
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.layer1 = nn.Sequential(
            nn.Conv2d(n_channels, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(nn.Dropout(0.5), nn.Linear(50_176, 4096), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(4096, n_classes), nn.Softmax(dim=1))

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

    def __init__(self, in_channels: int, out_channels: int):
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


class UNet(nn.Module): #Fix issue where input_channel parameter = 1
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
        for i, feature in enumerate(features):
            if i == 0:
                self.dc.append(DoubleConvolution(in_channels, feature))
            else:
                self.dc.append(DoubleConvolution(feature, 2 * feature))
            in_channels = feature

        # Calculate the Up Convolutions
        for feature in reversed(features):
            self.uc.append(
                nn.ConvTranspose2d(2 * feature, feature, kernel_size=2, stride=2)
            )
            self.uc.append(DoubleConvolution(2 * feature, feature))

        self.bottleneck = DoubleConvolution(features[-1], 2 * features[-1])
        self.final_convolution = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        """Forward pass of u-net."""
        skip_connections = list()
        for down in self.dc:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.uc), 2):
            x = self.uc[idx](x)
            skip_connections = skip_connections[idx // 2]

            if x.shape != skip_connections.shape:
                x = TF.resize(x, size=skip_connections.shape[2:])

            concat_skip = torch.cat((skip_connections, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_convolution(x)


class TumorClassifier(nn.Module):
    """Tumor Classifier Module that uses both categorical data and image data.

    The machine learning model uses a combination of an image or
    scan in conjunction with categorical data contained within
    the dicom file.

    """

    def __init__(self, cat_input_length: int):
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
            self.catlinears.append(
                nn.Linear(int(cat_input_length), int(cat_input_length / 2))
            )
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

    def __init__(self, in_channels: int = 1, n_classes: int = 2):
        """Init the Class."""
        super(TutorialNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(250_000, 120)  # 7744
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)

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


class InceptionStem(nn.Module):
    """The Stem Section of the Inception Model Architecture."""

    def __init__(self, in_channels: int = 3):
        """Init the class."""
        super(InceptionStem, self).__init__()
        # Convolutions
        self.conv1 = nn.Conv2d(
            in_channels, 32, kernel_size=3, stride=2, padding="valid"
        )
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding="valid")
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=0)
        ## Split 1
        ### Branch 1 (0 Convolutions in total)
        ### Branch 2 (1 Convolutions in total)
        self.conv121 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding="valid")
        ## Split 2
        ### Branch 1 (2 Convolutions in total)
        self.conv211 = nn.Conv2d(128, 64, kernel_size=3, padding=0)
        self.conv212 = nn.Conv2d(64, 96, kernel_size=7, padding="valid")
        ### Branch 2 (4 Convolutions in total)
        self.conv221 = nn.Conv2d(128, 64, kernel_size=1, padding=0)
        self.conv222 = nn.Conv2d(64, 64, kernel_size=(7, 1), padding=0)
        self.conv223 = nn.Conv2d(64, 64, kernel_size=(1, 7), padding=0)
        self.conv224 = nn.Conv2d(64, 96, kernel_size=3, padding="valid")
        ## Split 3
        ### Branch 1 (1 Convolutions in total)
        self.conv311 = nn.Conv2d(192, 192, kernel_size=3, stride=2, padding="valid")
        ###Branch 2 (0 Convolutions in total)

        # Batch Normalizations
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn121 = nn.BatchNorm2d(64)
        self.bn211 = nn.BatchNorm2d(64)
        self.bn212 = nn.BatchNorm2d(96)
        self.bn221 = nn.BatchNorm2d(64)
        self.bn222 = nn.BatchNorm2d(64)
        self.bn223 = nn.BatchNorm2d(64)
        self.bn224 = nn.BatchNorm2d(96)
        self.bn311 = nn.BatchNorm2d(192)

        # Repeatable Layers
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        """Forward loop of the model."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        # Split 1
        ## Branch 1
        x0 = self.maxpool(x)
        ## Branch 2
        x1 = self.conv121(x)
        x1 = self.bn121(x1)
        x1 = self.relu(x1)
        # End of Split 1
        x = torch.cat((x0, x1), dim=1)
        # Split 2
        ## Branch 1
        x0 = self.conv211(x)
        x0 = self.bn211(x0)
        x0 = self.relu(x0)
        x0 = self.conv212(x0)
        x0 = self.bn212(x0)
        x0 = self.relu(x0)
        ## Branch 2
        x1 = self.conv221(x)
        x1 = self.bn221(x1)
        x1 = self.relu(x1)
        x1 = self.conv222(x1)
        x1 = self.bn222(x1)
        x1 = self.relu(x1)
        x1 = self.conv223(x1)
        x1 = self.bn223(x1)
        x1 = self.relu(x1)
        x1 = self.conv224(x1)
        x1 = self.bn224(x1)
        x1 = self.relu(x1)
        # End of Split 2
        x = torch.cat((x0, x1), dim=1)
        # Split 3
        ## Branch 1
        x0 = self.conv311(x)
        x0 = self.bn311(x0)
        x0 = self.relu(x0)
        ## Branch 2
        x1 = self.maxpool(x)
        x = torch.cat((x0, x1), dim=1)
        return x


class InceptionA(nn.Module):
    """The First Inception Block Within the Inception Network."""

    def __init__(self, n_features: int):
        """Init the class."""
        super(InceptionA, self).__init__()

        # Convolutions
        ## Branch 1 (1 in total)
        self.conv11 = nn.Conv2d(n_features, 96, kernel_size=1, padding=0)
        ## Branch 2 (1 in total)
        self.conv21 = nn.Conv2d(n_features, 96, kernel_size=3, stride=2, padding=0)
        ## Branch 3 (2 in total)
        self.conv31 = nn.Conv2d(n_features, 64, kernel_size=1, padding=0)
        self.conv32 = nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=0)
        ## Branch 4 (3 in total)
        self.conv41 = nn.Conv2d(n_features, 64, kernel_size=1, padding=0)
        self.conv42 = nn.Conv2d(64, 96, kernel_size=1, padding=0)
        self.conv43 = nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=0)

        # Batch Normalizations
        self.bn11 = nn.BatchNorm2d(96)
        self.bn21 = nn.BatchNorm2d(96)
        self.bn31 = nn.BatchNorm2d(64)
        self.bn32 = nn.BatchNorm2d(96)
        self.bn41 = nn.BatchNorm2d(64)
        self.bn42 = nn.BatchNorm2d(96)
        self.bn43 = nn.BatchNorm2d(96)

        # Repeatable Layers
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward loop of model."""
        # Branch 1
        x1 = self.avgpool(x)
        x1 = self.conv11(x1)
        x1 = self.bn11(x1)
        x1 = self.relu(x1)
        # Branch 2
        x2 = self.conv21(x)
        x2 = self.bn21(x2)
        x2 = self.relu(x2)
        # Branch 3
        x3 = self.conv31(x)
        x3 = self.bn31(x3)
        x3 = self.relu(x3)
        x3 = self.conv32(x3)
        x3 = self.bn32(x3)
        x3 = self.relu(x3)
        # Branch 4
        x4 = self.conv41(x)
        x4 = self.bn41(x4)
        x4 = self.relu(x4)
        x4 = self.conv42(x4)
        x4 = self.bn42(x4)
        x4 = self.relu(x4)
        x4 = self.conv43(x4)
        x4 = self.bn43(x4)
        x4 = self.relu(x4)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        return x


class InceptionB(nn.Module):
    """The Second Inception Block Within the Inception Network."""

    def __init__(self, n_features: int):
        """Init the class."""
        super(InceptionB, self).__init__()
        # Convolutions
        ## Branch 1 (1 in total)
        self.conv11 = nn.Conv2d(n_features, 128, kernel_size=1, padding=0)
        ## Branch 2 (1 in total)
        self.conv21 = nn.Conv2d(n_features, 384, kernel_size=1, padding=0)
        ## Branch 3 (3 in total)
        self.conv31 = nn.Conv2d(n_features, 192, kernel_size=1, padding=0)
        self.conv32 = nn.Conv2d(192, 224, kernel_size=(1, 7), padding=0)
        self.conv33 = nn.Conv2d(224, 256, kernel_size=(1, 7), padding=(0, 6))
        ## Branch 4 (5 in total)
        self.conv41 = nn.Conv2d(n_features, 192, kernel_size=1, padding=0)
        self.conv42 = nn.Conv2d(192, 192, kernel_size=(1, 7), padding=(0, 0))
        self.conv43 = nn.Conv2d(192, 224, kernel_size=(7, 1), padding=(0, 0))
        self.conv44 = nn.Conv2d(224, 224, kernel_size=(1, 7), padding=(0, 6))
        self.conv45 = nn.Conv2d(224, 256, kernel_size=(7, 1), padding=(6, 0))

        # Batch Normalizations
        self.bn11 = nn.BatchNorm2d(128)
        self.bn21 = nn.BatchNorm2d(384)
        self.bn31 = nn.BatchNorm2d(192)
        self.bn32 = nn.BatchNorm2d(224)
        self.bn33 = nn.BatchNorm2d(256)
        self.bn41 = nn.BatchNorm2d(192)
        self.bn42 = nn.BatchNorm2d(192)
        self.bn43 = nn.BatchNorm2d(224)
        self.bn44 = nn.BatchNorm2d(224)
        self.bn45 = nn.BatchNorm2d(256)

        # Repeatable Layers
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward Loop of the Module."""
        # Branch 1
        x1 = self.avgpool(x)
        x1 = self.conv11(x1)
        x1 = self.bn11(x1)
        x1 = self.relu(x1)
        # Branch 2
        x2 = self.conv21(x)
        x2 = self.bn21(x2)
        x2 = self.relu(x2)
        # Branch 3
        x3 = self.conv31(x)
        x3 = self.bn31(x3)
        x3 = self.relu(x3)
        x3 = self.conv32(x3)
        x3 = self.bn32(x3)
        x3 = self.relu(x3)
        x3 = self.conv33(x3)
        x3 = self.bn33(x3)
        x3 = self.relu(x3)
        # Branch 4
        x4 = self.conv41(x)
        x4 = self.bn41(x4)
        x4 = self.relu(x4)
        x4 = self.conv42(x4)
        x4 = self.bn42(x4)
        x4 = self.relu(x4)
        x4 = self.conv43(x4)
        x4 = self.bn43(x4)
        x4 = self.relu(x4)
        x4 = self.conv44(x4)
        x4 = self.bn44(x4)
        x4 = self.relu(x4)
        x4 = self.conv45(x4)
        x4 = self.bn45(x4)
        x4 = self.relu(x4)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        return x


class InceptionC(nn.Module):
    """The Third Inception Block Within the Inception Network."""

    def __init__(self, n_features: int):
        """Init the class."""
        super(InceptionC, self).__init__()
        # Convolutions
        ## Branch 1 (1 in total)
        self.conv11 = nn.Conv2d(n_features, 256, kernel_size=1, padding=0)
        ## Branch 2 (1 in total)
        self.conv21 = nn.Conv2d(n_features, 256, kernel_size=1, stride=2, padding=0)
        ## Branch 3 (3 in total)
        self.conv31 = nn.Conv2d(n_features, 384, kernel_size=1, padding=0)
        self.conv32l = nn.Conv2d(384, 256, kernel_size=(1, 3), stride=2, padding=(0, 1))
        self.conv32r = nn.Conv2d(384, 256, kernel_size=(3, 1), stride=2, padding=(1, 0))
        ## Branch 4 (5 in total)
        self.conv41 = nn.Conv2d(n_features, 384, kernel_size=1, padding=0)
        self.conv42 = nn.Conv2d(384, 448, kernel_size=(1, 3), padding=0)
        self.conv43 = nn.Conv2d(448, 512, kernel_size=(3, 1), padding=0)
        self.conv44l = nn.Conv2d(512, 256, kernel_size=(3, 1), stride=2, padding=(2, 1))
        self.conv44r = nn.Conv2d(512, 256, kernel_size=(1, 3), stride=2, padding=(1, 2))

        # Batch Normalizations
        self.bn11 = nn.BatchNorm2d(256)
        self.bn21 = nn.BatchNorm2d(256)
        self.bn31 = nn.BatchNorm2d(384)
        self.bn32l = nn.BatchNorm2d(256)
        self.bn32r = nn.BatchNorm2d(256)
        self.bn41 = nn.BatchNorm2d(384)
        self.bn42 = nn.BatchNorm2d(448)
        self.bn43 = nn.BatchNorm2d(512)
        self.bn44l = nn.BatchNorm2d(256)
        self.bn44r = nn.BatchNorm2d(256)

        # Repeatable Layers
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass of the network."""
        # Branch 1
        x1 = self.avgpool(x)
        x1 = self.conv11(x1)
        x1 = self.bn11(x1)
        x1 = self.relu(x1)
        # Branch 2
        x2 = self.conv21(x)
        x2 = self.bn21(x2)
        x2 = self.relu(x2)
        # Branch 3
        x3 = self.conv31(x)
        x3 = self.bn31(x3)
        x3 = self.relu(x3)
        x3l = self.conv32l(x3)
        x3l = self.bn32l(x3l)
        x3l = self.relu(x3l)
        x3r = self.conv32r(x3)
        x3r = self.bn32r(x3r)
        x3r = self.relu(x3r)
        # Branch 4
        x4 = self.conv41(x)
        x4 = self.bn41(x4)
        x4 = self.relu(x4)
        x4 = self.conv42(x4)
        x4 = self.bn42(x4)
        x4 = self.relu(x4)
        x4 = self.conv43(x4)
        x4 = self.bn43(x4)
        x4 = self.relu(x4)
        x4l = self.conv44l(x4)
        x4l = self.bn44l(x4l)
        x4l = self.relu(x4l)
        x4r = self.conv44r(x4)
        x4r = self.bn44r(x4r)
        x4r = self.relu(x4r)
        x = torch.cat((x1, x2, x3l, x3r, x4l, x4r), dim=1)
        return x


class ReductionA(nn.Module):
    """First Reduction Block from the Inception Neural Network V4."""

    def __init__(self, n_features: int):
        """Init the class."""
        super(ReductionA, self).__init__()
        # Convolutions
        ## Branch 1 (0 in total)
        ## Branch 2 (1 in total)
        self.conv21 = nn.Conv2d(
            n_features, 384, kernel_size=3, stride=2, padding="valid"
        )
        ## Branch 3 (3 in total)
        self.conv31 = nn.Conv2d(n_features, 192, kernel_size=1, padding=0)
        self.conv32 = nn.Conv2d(192, 224, kernel_size=1, padding=0)
        self.conv33 = nn.Conv2d(224, 256, kernel_size=3, stride=2, padding="valid")

        # Batch Normalizations
        self.bn21 = nn.BatchNorm2d(384)
        self.bn31 = nn.BatchNorm2d(192)
        self.bn32 = nn.BatchNorm2d(224)
        self.bn33 = nn.BatchNorm2d(256)

        # Repeatable Layers
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass of the neural network."""
        x1 = self.maxpool(x)
        x2 = self.conv21(x)
        x2 = self.bn21(x2)
        x2 = self.relu(x2)
        x3 = self.conv31(x)
        x3 = self.bn31(x3)
        x3 = self.relu(x3)
        x3 = self.conv32(x3)
        x3 = self.bn32(x3)
        x3 = self.relu(x3)
        x3 = self.conv33(x3)
        x3 = self.bn33(x3)
        x3 = self.relu(x3)
        x = torch.cat((x1, x2, x3), dim=1)
        return x


class ReductionB(nn.Module):
    """The Second Reduction From the Inception V4 Neural Network."""

    def __init__(self, n_features: int):
        """Init the class."""
        super(ReductionB, self).__init__()
        # Convolutions
        ## Branch 1 (0 in total)
        ## Branch 2 (2 in total)
        self.conv21 = nn.Conv2d(n_features, 192, kernel_size=1, padding=0)
        self.conv22 = nn.Conv2d(192, 192, kernel_size=3, stride=2, padding="valid")
        ### Branch 3 (4 in total)
        self.conv31 = nn.Conv2d(n_features, 256, kernel_size=1, padding=0)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=(1, 7), padding=(0, 3))
        self.conv33 = nn.Conv2d(256, 320, kernel_size=(7, 1), padding=(3, 0))
        self.conv34 = nn.Conv2d(320, 320, kernel_size=3, stride=2, padding="valid")

        # Batch Normalizations
        self.bn21 = nn.BatchNorm2d(192)
        self.bn22 = nn.BatchNorm2d(192)
        self.bn31 = nn.BatchNorm2d(256)
        self.bn32 = nn.BatchNorm2d(256)
        self.bn33 = nn.BatchNorm2d(320)
        self.bn34 = nn.BatchNorm2d(320)

        # Repeatable Layers
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        """Forward pass of the neural network."""
        x1 = self.maxpool(x)
        x2 = self.conv21(x)
        x2 = self.bn21(x2)
        x2 = self.relu(x2)
        x2 = self.conv22(x2)
        x2 = self.bn22(x2)
        x2 = self.relu(x2)
        x3 = self.conv31(x)
        x3 = self.bn31(x3)
        x3 = self.relu(x3)
        x3 = self.conv32(x3)
        x3 = self.bn32(x3)
        x3 = self.relu(x3)
        x3 = self.conv33(x3)
        x3 = self.bn33(x3)
        x3 = self.relu(x3)
        x3 = self.conv34(x3)
        x3 = self.bn34(x3)
        x3 = self.relu(x3)
        x = torch.cat((x1, x2, x3), dim=1)
        return x


class InceptionV4(nn.Module):
    """The Completed Inception Model."""

    def __init__(self, n_classes: int, n_channels: int):
        """Init the class."""
        super(InceptionV4, self).__init__()
        self.stem = InceptionStem(n_channels)
        self.ia = InceptionA(384)
        self.ra = ReductionA(384)
        self.ib = InceptionB(1024)
        self.rb = ReductionB(1024)
        self.ic = InceptionC(1536)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=0)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(1536, 1000)
        self.linear2 = nn.Linear(1000, 500)
        self.linear3 = nn.Linear(500, 250)
        self.linear4 = nn.Linear(250, 100)
        self.linear5 = nn.Linear(100, n_classes)
        self.softmax = nn.Softmax(dim=1)

        # Repeatable layer(s)
        self.dropout = nn.Dropout(0.8)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass of network."""
        x = self.stem(x)
        x = self.ia(x)
        x = self.ra(x)
        x = self.ib(x)
        x = self.rb(x)
        x = self.ic(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.relu(x)
        x = self.linear5(x)
        x = self.softmax(x)
        return x


if __name__ == "__main__":
    _main()

# websites:
# https://arxiv.org/pdf/1311.2901.pdf
# https://vitalflux.com/different-types-of-cnn-architectures-explained-examples/
