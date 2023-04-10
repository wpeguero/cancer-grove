"""Models Module
-----------------

This file will contain all of the actualized models
created from the abstract model class(es) made within
the base.py file."""
import os
from tensorflow.keras.layers import Conv2D, Dense, Rescaling, Flatten, MaxPooling2D, Dropout, RandomZoom, Input, Concatenate, BatchNormalization
from tensorflow.keras.optimizers.experimental import Adagrad
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, AUC
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model, split_dataset
from tensorflow.keras.models import save_model
import tensorflow as tf
from src.pipeline import load_training_data
import pandas as pd

tsize = 1_500
BATCH_SIZE = 4
validate = False
version=17

def _main():
    pass


def base_image_classifier(img_height:float, img_width:float):
    """Basic Image Classifier for model comparison improvement.

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
    """Base Tumor Classification Algorithm.

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


if __name__ == "__main__":
    _main()

#websites:
# https://arxiv.org/pdf/1311.2901.pdf
# https://vitalflux.com/different-types-of-cnn-architectures-explained-examples/
