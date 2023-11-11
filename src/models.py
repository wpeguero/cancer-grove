"""Models Module.

-----------------

This file will contain all of the actualized models
created from the abstract model class(es) made within
the base.py file.
"""
import os
import tracemalloc
import re

# Current issue: Loss is not working properly during training process
from keras.layers import Conv2D, Conv2DTranspose, Dense, Rescaling, Flatten, MaxPool2D, Dropout, Input, Concatenate, BatchNormalization, Resizing
from tensorflow.keras.optimizers.experimental import Adagrad
from keras.losses import SparseCategoricalCrossentropy, KLDivergence, BinaryCrossentropy
from keras.metrics import BinaryAccuracy, AUC, MeanIoU
from keras.models import Model, save_model
from keras.utils import plot_model, split_dataset
import tensorflow as tf
import torch
from torch import nn
import torchvision.transforms.functional as TF
import numpy as np
import pandas as pd

from pipeline import load_training_data, load_image, merge_dictionaries
from losses import Dice

img_size = (512, 512)
mask_size = (512, 512)
tsize = 8
BATCH_SIZE = 4
validate = False
version=3

def _main():
    tracemalloc.start()
    filepath = "data/Dataset_BUSI_with_GT/"
    filepath_dirs = os.listdir(filepath)
    path__malignant_images = filepath + filepath_dirs[1]
    path__benign_images = filepath + filepath_dirs[0]
    path__normal_images = filepath + filepath_dirs[2]
    malignant_images = os.listdir(path__malignant_images)
    benign_images= os.listdir(path__benign_images)
    normal_images= os.listdir(path__normal_images)
    # Get relative paths
    paths__malignant = [ path__malignant_images + '/' + image for image in malignant_images ]
    paths__benign = [ path__benign_images + '/' + image for image in benign_images]
    paths__normal = [ path__normal_images + '/' + image for image in normal_images]
    # Separate paths based on mask images and non-mask images
    paths__malignant_images = [ malignant_path for malignant_path in paths__malignant if 'mask' not in malignant_path ]
    paths__malignant_mask = [ malignant_path for malignant_path in paths__malignant if 'mask' in malignant_path ]
    paths__benign_images = [ benign_path for benign_path in paths__benign if 'mask' not in benign_path ]
    paths__benign_mask = [ benign_path for benign_path in paths__benign if 'mask' in benign_path ]
    paths__normal_images = [ normal_path for normal_path in paths__normal if 'mask' not in normal_path ]
    paths__normal_mask = [ normal_path for normal_path in paths__normal if 'mask' in normal_path ]
    # Collect the Images in dictionaries
    malignant_image_set = { str("m" + re.findall(r'\d+', mfile)[0]):load_image(mfile, img_size) for mfile in paths__malignant_images }
    malignant_mask_set = { str("m" + re.findall(r'\d+', mfile)[0]):load_image(mfile, mask_size) for mfile in paths__malignant_mask }
    benign_image_set = { str("b" + re.findall(r'\d+', mfile)[0]):load_image(mfile, img_size) for mfile in paths__benign_images }
    benign_mask_set = { str("b" + re.findall(r'\d+', mfile)[0]):load_image(mfile, mask_size) for mfile in paths__benign_mask }
    normal_image_set = { str("n" + re.findall(r'\d+', mfile)[0]):load_image(mfile, img_size) for mfile in paths__normal_images }
    normal_mask_set = { str("n" + re.findall(r'\d+', mfile)[0]):load_image(mfile, mask_size) for mfile in paths__normal_mask }
    # Merge images and mask
    malignant_dictionary = merge_dictionaries(malignant_image_set, malignant_mask_set)
    benign_dictionary = merge_dictionaries(benign_image_set, benign_mask_set)
    normal_dictionary = merge_dictionaries(normal_image_set, normal_mask_set)
    # Convert the values into input and output
    malignant_set = list()
    for key, value in malignant_dictionary.items():
        malignant_set.append({'id': key, 'image': value[0], 'mask': value[1]})
    df__malignant = pd.DataFrame(malignant_set)
    benign_set = list()
    for key, value in benign_dictionary.items():
        benign_set.append({'id': key, 'image': value[0], 'mask': value[1]})
    df__benign = pd.DataFrame(benign_set)
    normal_set = list()
    for key, value in normal_dictionary.items():
        normal_set.append({'id': key, 'image': value[0], 'mask': value[1]})
    df__normal = pd.DataFrame(normal_set)
    # Allocate entire dataset into a singular DataFrame
    df_set = pd.concat([df__malignant, df__benign], axis=0) #Excluded normal masks as they are all black
    # Get Images and Masks
    df = df_set.sample(frac=0.7, random_state=42)
    images = df['image'].tolist()
    images = np.asarray(images).astype('float32')
    masks = df['mask'].tolist()
    masks = np.asarray(masks).astype('int32')
    inputs, outputs = u_net(512, 512)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='Adam', loss=BinaryCrossentropy(), metrics=['accuracy', AUC(), BinaryAccuracy()])
    dataset = tf.data.Dataset.from_tensor_slices((images, masks)).batch(BATCH_SIZE)
    #dataset = dataset.shuffle(buffer_size=10).prefetch(tf.data.AUTOTUNE)
    cp_path = "models/weights/u_net{}.ckpt".format(version)
    cp_dir = os.path.dirname(cp_path)
    print("\nStarting Training\n")
    thistory = model.fit(dataset, epochs=100)
    save_model(model, './models/u_net{}'.format(version))
    hist_df = pd.DataFrame(thistory.history)
    hist_df.to_csv("data/history_unet{}.csv".format(version))
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    print("[ Top 10 ]")
    for stat in top_stats[:10]:
        print(stat)
    exit()


class BasicImageClassifier(nn.Module):
    """Create Basic Image Classifier for model comparison improvement.

    ...

    A class containing a simple classifier for any
    sort of image. The models stemming from this
    class will function to only classify the image
    in one manner alone (malignant or non-malignant).
    This model will not contain any rescaling or
    data augmentation to show how significant the
    accuracy between a model with rescaling and
    data augmentation is against a model without
    any of these.

    Parameters
    -----------
    img_height : Integer
        The height, in pixels, of the input images.
        This can be the maximum height of all images
        within the dataset to fit a varied amount
        that is equal or less than the declared height.

    img_width : Integer
        The width, in pixels, of the input images.
        This can also be the maximum width of all
        images within the dataset to fit a varied
        amount that is equal or smaller in width
        to the declared dimension.

    batch_size : Integer
        One of the factors of the total sample size.
        This is done to better train the model without
        allowing the model to memorize the data.

    Returns
    -------
    inputs : {img_input, cat_input}
        Input layers set to receive both image and
        categorical data. The image input contains
        images in the form of a 2D numpy array. The
        categorical input is a 1D array containing
        patient information. This is mainly comprised
        of categorical data, but some nominal data.

    x : Dense Layer
        The last layer of the model developed. As
        the model is fed through as the input of
        the next layer, the last layer is required
        to create the model using TensorFlow's Model
        class.
    """

    def __init__(self, img_height:int, img_width:int):
        """Initialize the image classifier."""
        super(BasicImageClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 96, stride=2)
        self.mp1 = nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        self.bn1 = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96, 256, stride=2)
        self.mp2 = nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 384)
        self.conv4 = nn.Conv2d(384, 256)
        self.mp3 = nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(4096, 4096)
        self.dropout = nn.Dropout(0.25)
        self.linear2 = nn.Linear(4096, 1000)
        self.linear3 = nn.Linear(1000, 500)
        self.linear4 = nn.Linear(500, 250)
        self.linear5 = nn.Linear(250, 100)
        self.linear6 = nn.Linear(100, 50)
        self.linear7 = nn.Linear(50, 25)
        self.linear8 = nn.Linear(25, 12)
        self.linear9 = nn.Linear(12, 2)

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
    """
    Does the Double Convolution shown within a unit of the U-Net.

    Parameter(s)
    ------------
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
    """
    Creates a U-Net model for image segmentation.

    Unique class built to develop U-Net models. Inherits from the
    Module class found in pytorch.

    Parameter(s)
    ------------
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
    """
    Complete Tumor Classification Algorithm.

    ...

    A class containing a simple classifier for any
    sort of image. The models stemming from this class
    will include rescaling and data augmentation
    for the sake and purpose of normalizing the data.

    Parameters
    -----------
    img_height : float
        The height, in pixels, of the input images.
        This can be the maximum height of all images
        within the dataset to fit a varied amount
        that is equal or less than the declared height.

    img_width : float
        The width, in pixels, of the input images.
        This can also be the maximum width of all
        images within the dataset to fit a varied
        amount that is equal or smaller in width
        to the declared dimension.
    """

    def __init__(self, img_height, img_width, num_cats):
        """Inialize the Model."""
        self.conv1 = nn.Conv2d()


if __name__ == "__main__":
    _main()

#websites:
# https://arxiv.org/pdf/1311.2901.pdf
# https://vitalflux.com/different-types-of-cnn-architectures-explained-examples/
