"""Models Module.

-----------------

This file will contain all of the actualized models
created from the abstract model class(es) made within
the base.py file.
"""
import os
import tracemalloc
import re

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Rescaling, Flatten, MaxPool2D, Dropout, Input, Concatenate, BatchNormalization, Resizing
from tensorflow.keras.optimizers.experimental import Adagrad
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, AUC
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model, split_dataset
from tensorflow.keras.models import save_model
import tensorflow as tf
import pandas as pd

from pipeline import load_training_data, load_image

tsize = 8
BATCH_SIZE = 2
validate = False
version=1

def _main():
    tracemalloc.start()
    filepath = "data/Dataset_BUSI_with_GT/"
    filepath_dirs = os.listdir(filepath)
    path__malignant_images = filepath + filepath_dirs[0]
    list__malim = os.listdir(path__malignant_images)
    list_malim_paths = [ path__malignant_images + '/' + malim for malim in list__malim ]
    paths__malignant_masks = [ malim_path for malim_path in list_malim_paths if 'mask' in malim_path ]
    paths__malignant_images = [ malim_path for malim_path in list_malim_paths if 'mask' not in malim_path ]
    path__benign_images = filepath + filepath_dirs[1]
    list__benim = os.listdir(path__benign_images)
    list_benim_paths = [ path__benign_images + '/' + benim for benim in list__benim ]
    paths__benign_masks = [ benim_path for benim_path in list_benim_paths if 'mask' in benim_path ]
    paths__benign_images = [ benim_path for benim_path in list_benim_paths if 'mask' not in benim_path ]
    path__normal_images = filepath + filepath_dirs[2]
    list__norim = os.listdir(path__normal_images)
    list_norim_paths = [ path__normal_images + '/' + norim for norim in list__norim ]
    paths__normal_masks = [ norim_path for norim_path in list_norim_paths if 'mask' in norim_path ]
    paths__normal_images = [ norim_path for norim_path in list_norim_paths if 'mask' not in norim_path ]
    malignant_image_set = [ {re.findall(r'\d+', mfile)[0]:load_image(mfile)} for mfile in paths__malignant_images ]
    malignant_mask_set = [ {re.findall(r'\d+', mfile)[0]:load_image(mfile)} for mfile in paths__malignant_masks ]
    benign_image_set = [ {re.findall(r'\d+', mfile)[0]:load_image(mfile)} for mfile in paths__benign_images ]
    benign_mask_set = [ {re.findall(r'\d+', bfile)[0]:load_image(bfile)} for bfile in paths__benign_masks ]
    normal_image_set = [ {re.findall(r'\d+', nfile)[0]:load_image(nfile)} for nfile in paths__normal_images ]
    normal_mask_set = [ {re.findall(r'\d+', nfile)[0]:load_image(nfile)} for nfile in paths__normal_masks ]
    print(normal_image_set[0])
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    print("[ Top 10 ]")
    for stat in top_stats[:10]:
        print(stat)
    exit()
    inputs, outputs = u_net()
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='Adam', loss=CategoricalCrossentropy(from_logits=False), metrics=[CategoricalAccuracy(), AUC(from_logits=False)], run_eagerly=True)
    #plot_model(model, show_shapes=True, to_file='./u_net{}.png'.format(version))
    dataset = tf.data.Dataset.from_tensor_slices()
    dataset = dataset.shuffle(buffer_size=10).prefetch(tf.data.AUTOTUNE)
    cp_path = "models/weights/u_net{}.ckpt".format(version)
    cp_dir = os.path.dirname(cp_path)
    thistory = model.fit(dataset, epochs=50)
    save_model(model, './models/u_net{}'.format(version))
    hist_df = pd.DataFrame(thistory.history)
    hist_df.to_csv("data/history_unet{}.csv".format(version))


def base_image_classifier(img_height:float, img_width:float):
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

    batch_size : int
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
    img_input = Input(shape=(img_height,img_width,1), name="image")
    # Set up the images
    x = Conv2D(96, 7, strides=(2,2), activation='relu')(img_input)
    x = MaxPooling2D(pool_size=(3,3), strides=2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, 5, strides=(2,2), activation='relu')(x)
    x = MaxPooling2D(pool_size=(3,3), strides=2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(384, 3, padding='same', activation='relu')(x)
    #x = Dropout(0.3)(x)
    x = Conv2D(384, 3, padding='same', activation='relu')(x)
    x = Conv2D(256, 3, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(3,3), strides=2)(x)
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dense(1000, activation='relu')(x)
    x = Dense(500, activation='relu')(x)
    x = Dense(250, activation='relu')(x)
    x = Dense(100, activation='relu')(x)
    x = Dense(50, activation='relu')(x)
    x = Dense(25, activation='relu')(x)
    x = Dense(12, activation='relu')(x)
    x = Dense(2, activation='sigmoid')(x)
    return img_input, x

def base_tumor_classifier(img_height:float, img_width:float):
    """Create Base Tumor Classification Algorithm.

    ...

    A class containing a simple classifier for side-view
    image. The models stemming from this class
    will include rescaling for the sake and purpose
    of normalizing the data.

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

    Returns
    -------
    inputs : {img_input, cat_input}
        Input layers set to receive both image and
        categorical data. The image input contains
        images in the form of a 2D numpy array. The
        categorical input is a 1D array containing
        patient information. This is mainly comprised
        of categorical data, but some nominal data.

    output : Dense Layer
        The last layer of the model developed. As
        the model is fed through as the input of
        the next layer, the last layer is required
        to create the model using TensorFlow's Model
        class.
    """
    img_input = Input(shape=(img_height, img_width, 1), name='image')
    cat_input = Input(shape=(2), name='cat')
    inputs = [img_input, cat_input]
    # Set up the images
    x = Rescaling(1./255, input_shape=(img_height, img_width,1))(img_input)
    x = Conv2D(96, 3, strides=(2,2), activation='relu')(x)
    x = MaxPooling2D(pool_size=(3,3), strides=2)(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dropout(0.3)(x)
    x = Dense(250, activation='relu')(x)
    x = Dense(100, activation='relu')(x)
    x = Dense(50, activation='relu')(x)
    x = Dense(25, activation='relu')(x)
    #Set up the categorical data
    y = Dense(2, activation='relu')(cat_input)
    y = Dense(1, activation='relu')(y)
    # Merge both layers

    together = Concatenate(axis=1)([x,y])
    together = Dense(13, activation='relu')(together)
    output = Dense(2, activation='sigmoid', name='class')(together)
    return inputs, output

def tumor_classifier(img_height:float, img_width:float):
    """Complete Tumor Classification Algorithm.

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
    
    batch_size : int *
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
    
    output : Dense Layer
        The last layer of the model developed. As
        the model is fed through as the input of
        the next layer, the last layer is required
        to create the model using TensorFlow's Model
        class.
    
    ---
    
    *Deprecated
    """
    img_input = Input(shape=(img_height, img_width, 1), name='image')
    cat_input = Input(shape=(2), name='cat')
    inputs = [img_input, cat_input]
    # Set up the images
    x = Rescaling(1./255, input_shape=(img_height, img_width,1))(img_input)
    x = Conv2D(64*5, 3, padding='same', strides=(2,2), activation='relu')(x)
    x = Conv2D(64*5, 3, padding='same', strides=(2,2), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2,2), strides=2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(128*5, 5,padding='same',  strides=(2,2), activation='relu')(x)
    x = Conv2D(128*5, 5,padding='same',  strides=(2,2), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2,2), strides=2)(x)
    x = Conv2D(256*5, 5, padding='same', activation='relu')(x)
    x = Conv2D(256*5, 5, padding='same', activation='relu')(x)
    x = Conv2D(256*5, 5, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2,2), strides=2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(512*5, 5, padding='same', activation='relu')(x)
    x = Conv2D(512*5, 5, padding='same', activation='relu')(x)
    x = Conv2D(512*5, 5, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2,2), strides=2)(x)
    x = Conv2D(512*5, 5, padding='same', activation='relu')(x)
    x = Conv2D(512*5, 5, padding='same', activation='relu')(x)
    x = Conv2D(512*5, 5, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2), strides=2)(x)
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dense(4096, activation='relu')(x)
    x = Dense(1000, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(500, activation='relu')(x)
    x = Dense(250, activation='relu')(x)
    x = Dense(100, activation='relu')(x)
    x = Dense(50, activation='relu')(x)
    #x = Dropout(0.5)(x)
    x = Dense(25, activation='relu')(x)
    #Set up the categorical data
    y = Dense(2, activation='relu')(cat_input)
    y = Dense(1, activation='relu')(y)
    # Merge both layers

    together = Concatenate(axis=1)([x,y])
    together = Dense(13, activation='relu')(together)
    output = Dense(2, activation='sigmoid', name='class')(together)
    return inputs, output

def u_net(img_height:int, img_width:int):
    """Create UNet Model.

    ---------------------

    This function is used to develop a U Neural Network.
    This model works by using Convolutional Neural Networks
    and skipping to the end of the model in a U pattern to
    prevent data loss and retain certain information about
    the image.

    Parameter(s)
    ------------

    img_height : integer
        The height of the image.

    img_width : integer
        The width of the image.

    Returns
    -------
    img_output : TensorTlow Model
    img_input : TensorFlow Input
    """
    img_input = Input(shape=(img_height, img_width, 1), name='image')
    # First Convolutional Block
    encx1 = Conv2D(64, (3,3), activation='relu')(img_input)
    encx1 = Conv2D(64, (3,3), activation='relu')(encx1)
    enc1 = MaxPool2D((2,2))(encx1)

    # Second Convolutional Block
    encx2 = Conv2D(128, (3,3), activation='relu')(enc1)
    encx2 = Conv2D(128, (3,3), activation='relu')(encx2)
    enc2 = MaxPool2D((2,2))(encx2)

    # Third Convolutional Block
    encx3 = Conv2D(256, (3,3), activation='relu')(enc2)
    encx3 = Conv2D(256, (3,3), activation='relu')(encx3)
    enc3 = MaxPool2D((2,2))(encx3)

    # Fourth Convolutional Block
    encx4 = Conv2D(512, (3,3), activation='relu')(enc3)
    encx4 = Conv2D(512, (3,3), activation='relu')(encx4)
    enc4 = MaxPool2D((2,2))(encx4)
    # Fifth Convolutional Block
    encx5 = Conv2D(1024, (3,3), activation='relu')(enc4)
    encx5 = Conv2D(1024, (3,3), activation='relu')(encx5)

    # Sixth Convolutional Block
    decx6 = Conv2DTranspose(512, (2,2), strides=2, activation='relu')(encx5)
    rencx4 = tf.image.resize(encx4,[decx6.shape[1], decx6.shape[2]])
    concat = Concatenate(axis=-1)([rencx4, decx6])
    decx6 = Conv2D(512, (3,3), activation='relu')(concat)
    decx6 = Conv2D(512, (3,3), activation='relu')(decx6)

    # Seventh Convolutional Block
    decx7 = Conv2DTranspose(256, (2,2), strides=2, activation='relu')(decx6)
    rencx3 = tf.image.resize(encx4, [decx7.shape[1], decx7.shape[2]])
    concat = Concatenate(axis=-1)([decx7, rencx3])
    decx7 = Conv2D(256, (3,3), activation='relu')(concat)
    decx7 = Conv2D(256, (3,3), activation='relu')(decx7)

    # Eighth Convolutional Network
    decx8 = Conv2DTranspose(128, (2,2), activation='relu')(decx7)
    rencx2 = tf.image.resize(encx2, [decx8.shape[1], decx8.shape[2]])
    concat = Concatenate(axis=-1)([decx8, rencx2])
    decx8 = Conv2D(128, (3,3), activation='relu')(concat)
    decx8 = Conv2D(128, (3,3), activation='relu')(decx8)

    # Last convolutional Network
    decx9 = Conv2DTranspose(64, (2,2), activation='relu')(decx8)
    rencx1 = tf.image.resize(encx1, [decx9.shape[1], decx9.shape[2]])
    concat = Concatenate(axis=-1)([decx9, rencx1])
    decx9 = Conv2D(64, (3,3), activation='relu')(concat)
    decx9 = Conv2D(64, (3,3), activation='relu')(decx9)
    img_output = Conv2D(2, (1,1), activation='relu')(decx9)
    return img_input, img_output


if __name__ == "__main__":
    _main()

#websites:
# https://arxiv.org/pdf/1311.2901.pdf
# https://vitalflux.com/different-types-of-cnn-architectures-explained-examples/
