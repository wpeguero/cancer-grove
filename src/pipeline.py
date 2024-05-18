"""Algorithms used to process data before modeling.

A set of algorithms used to feed in and process
data before used within the model. This will contain
the data extraction from its rawest form and output
the final form of the data set. The main source of
data will be image related from the Cancer Imaging
Archive.
"""
import os
import pathlib
import json
from collections import defaultdict
import re

import numpy as np
import plotly.express as px
import polars as pl
from pydicom import dcmread
from PIL import Image
from pydicom.errors import InvalidDicomError
import torch
from torch import optim, nn
from torch.utils import data
from torchvision import datasets, transforms

from models import CustomCNN, AlexNet, InceptionStem, InceptionA, InceptionB, InceptionC, ReductionA, ReductionB, InceptionV4, TutorialNet
from datasets import DICOMSet
from trainers import Trainer, VERSION

img_size = (512, 512)
torch.manual_seed(42)
torch.cuda.manual_seed(42)


def _main():
    """Test the new functions."""
    dir_train = "data/images/"
    cat_trans = create_target_transform(2)
    img_set = datasets.ImageFolder(root=dir_train, transform=STANDARD_IMAGE_TRANSFORMS, target_transform=cat_trans)
    train_size = int(0.7*len(img_set))
    val_size = len(img_set) - train_size
    train_set, val_set = data.random_split(img_set, [train_size, val_size])
    dets = list()
    for i, (image, label) in enumerate(train_set):
        mean_pixel_intensity = torch.mean(image)
        dets.append(mean_pixel_intensity)
    img_mean = torch.mean(torch.Tensor(dets))
    img_std = torch.std(torch.Tensor(dets))
    print("mean: {}\nStandard Deviation: {}".format(img_mean, img_std))
    exit()
    train_loader = data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4)
    val_loader = data.DataLoader(val_set, batch_size=64, shuffle=True, num_workers=4)
    #Loading the models
    model1 = InceptionV4(2, 1)
    #Loading optimizers
    opt1 = optim.Adam(model1.parameters(), lr=0.003)
    #Loading the Losses
    loss1 = nn.BCELoss()
    #Loading the Trainers
    trainer1 = Trainer(model1, opt1, loss1)
    # Training and saving models
    trainer1.train(train_loader, 160, gpu=True)

    trained_model1 = trainer1.get_model()
    trainer1.test(trained_model1, val_loader, classes=('cat', 'loaf'),gpu=True)
    torch.save(trained_model1.state_dict(), "models/inceptionv4_catloaf.pt")

if __name__ == "__main__":
    _main()
