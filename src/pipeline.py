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
from typing import Optional
import re
import time

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
import utils
from stats import calculate_image_t_test

img_size = (512, 512)
torch.manual_seed(42)
torch.cuda.manual_seed(42)


def _main():
    """Test the new functions."""
    dir = 'data/CBIS-DDSM/images/'
    fpaths = utils.get_file_paths(dir, 'data/CBIS-DDSM/paths.csv')
    df__paths = pl.read_csv('data/CBIS-DDSM/paths.csv')
    pathset = list()
    for row in df__paths.iter_rows(named=True):
        fpath = row['paths']
        fpath = fpath[:-1]
        components = fpath.split('/')
        unique_id = components[3]
        if 'ROI mask' in fpath:
            label = 'mask'
        elif 'cropped images' in fpath:
            label = 'crop'
        elif 'full mammogram images' in fpath:
            label = 'full'
        else:
            label = ''
        pathset.append({'paths':fpath, 'UID':unique_id, 'img type':label})
    df__nupaths = pl.DataFrame(pathset)
    df__nupaths.write_csv('data/CBIS-DDSM/nupaths.csv')


if __name__ == "__main__":
    _main()
