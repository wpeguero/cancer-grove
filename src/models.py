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
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, AUC
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model, split_dataset
from tensorflow.keras.models import save_model
import tensorflow as tf
import numpy as np
import pandas as pd

from pipeline import load_training_data, load_image, merge_dictionaries

img_size = (512, 512)
mask_size = (324, 324)
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
    df_set = pd.concat([df__normal, df__malignant, df__benign], axis=0)
    # Get Images and Masks
    df = df_set.sample(frac=0.7, random_state=42)
    images = df['image'].tolist()
    images = np.asarray(images).astype('float32')
    masks = df['mask'].tolist()
    masks = np.asarray(masks).astype('float32')
    inputs, outputs = u_net(512, 512)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=[AUC(from_logits=False), 'accuracy'], run_eagerly=True)
    plot_model(model, show_shapes=True, to_file='./models/u_net{}.png'.format(version))
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
    rate = 0.3
    img_input = Input(shape=(img_height, img_width, 1), name='image')
    # First Convolutional Block
    encx1 = Conv2D(64, (3,3), activation='relu')(img_input)
    encx1 = Conv2D(64, (3,3), activation='relu')(encx1)
    enc1 = MaxPool2D((2,2))(encx1)
    enc1 = Dropout(rate)(enc1)

    # Second Convolutional Block
    encx2 = Conv2D(128, (3,3), activation='relu')(enc1)
    encx2 = Conv2D(128, (3,3), activation='relu')(encx2)
    enc2 = MaxPool2D((2,2))(encx2)
    enc2 = Dropout(rate)(enc2)

    # Third Convolutional Block
    encx3 = Conv2D(256, (3,3), activation='relu')(enc2)
    encx3 = Conv2D(256, (3,3), activation='relu')(encx3)
    enc3 = MaxPool2D((2,2))(encx3)
    enc2 = Dropout(rate)(enc3)

    # Fourth Convolutional Block
    encx4 = Conv2D(512, (3,3), activation='relu')(enc3)
    encx4 = Conv2D(512, (3,3), activation='relu')(encx4)
    enc4 = MaxPool2D((2,2))(encx4)
    enc4 = Dropout(rate)(enc4)

    # Fifth Convolutional Block
    encx5 = Conv2D(1024, (3,3), activation='relu')(enc4)
    encx5 = Conv2D(1024, (3,3), activation='relu')(encx5)

    # Sixth Convolutional Block
    decx6 = Conv2DTranspose(512, (2,2), strides=2, activation='relu')(encx5)
    encx4 = Dropout(rate)(encx4)
    rencx4 = tf.image.resize(encx4,[decx6.shape[1], decx6.shape[2]])
    concat = Concatenate(axis=-1)([rencx4, decx6])
    decx6 = Conv2D(512, (3,3), activation='relu')(concat)
    decx6 = Conv2D(512, (3,3), activation='relu')(decx6)

    # Seventh Convolutional Block
    decx7 = Conv2DTranspose(256, (2,2), strides=2, activation='relu')(decx6)
    encx3 = Dropout(rate)(encx3)
    rencx3 = tf.image.resize(encx3, [decx7.shape[1], decx7.shape[2]])
    concat = Concatenate(axis=-1)([decx7, rencx3])
    decx7 = Conv2D(256, (3,3), activation='relu')(concat)
    decx7 = Conv2D(256, (3,3), activation='relu')(decx7)

    # Eighth Convolutional Network
    decx8 = Conv2DTranspose(128, (2,2), strides=2, activation='relu')(decx7)
    encx2 = Dropout(rate)(encx2)
    rencx2 = tf.image.resize(encx2, [decx8.shape[1], decx8.shape[2]])
    concat = Concatenate(axis=-1)([decx8, rencx2])
    decx8 = Conv2D(128, (3,3), activation='relu')(concat)
    decx8 = Conv2D(128, (3,3), activation='relu')(decx8)

    # Last convolutional Network
    decx9 = Conv2DTranspose(64, (2,2), strides=2, activation='relu')(decx8)
    encx1 = Dropout(rate)(encx1)
    rencx1 = tf.image.resize(encx1, [decx9.shape[1], decx9.shape[2]])
    concat = Concatenate(axis=-1)([decx9, rencx1])
    decx9 = Conv2D(64, (3,3), activation='relu')(concat)
    decx9 = Conv2D(64, (3,3), activation='relu')(decx9)
    img_output = Conv2D(1, (1,1), activation='relu')(decx9)
    return img_input, img_output


if __name__ == "__main__":
    _main()

#websites:
# https://arxiv.org/pdf/1311.2901.pdf
# https://vitalflux.com/different-types-of-cnn-architectures-explained-examples/
