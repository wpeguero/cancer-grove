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

from models import CustomCNN, AlexNet, InceptionStem, InceptionA, InceptionB, InceptionC, ReductionA, ReductionB, InceptionV4
from datasets import DICOMSet
from trainers import TrainModel, VERSION
import models

img_size = (512, 512)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
STANDARD_IMAGE_TRANSFORMS = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(img_size, antialias=True),
        transforms.Grayscale(num_output_channels=1),
        transforms.Normalize((0.5), (0.5))
        ])

def _main():
    """Test the new functions."""
    fn__train = "data/CMMD-Set/valid_set.csv"
    train = pl.read_csv(fn__train) # Need to transform column to numerical representation
    # Transform categorical into numerical
    with open('data/CMMD-Set/label.json', 'r') as fp:
        label = json.load(fp)
        fp.close
    train = train.with_columns(pl.col('classification').replace(label, default=None))
    # Transform and load the datasets
    cat_trans = create_target_transform(2)
    train_set = DICOMSet(train, label_col='classification', img_col='path', image_transforms=STANDARD_IMAGE_TRANSFORMS, categorical_transforms=cat_trans)
    train_loader = data.DataLoader(train_set, shuffle=True, batch_size=32, num_workers=8)
    #Loading the models
    model1 = InceptionV4(2, 1)
    model2 = CustomCNN(1, 2)
    model3 = AlexNet(1, 2)
    #Loading optimizers
    opt1 = optim.SGD(model1.parameters(), lr=0.003, weight_decay=0.005, momentum=0.9)
    opt2 = optim.SGD(model2.parameters(), lr=0.003, weight_decay=0.005, momentum=0.9)
    opt3 = optim.SGD(model3.parameters(), lr=0.003, weight_decay=0.005, momentum=0.9)
    #Loading the Losses
    loss1 = nn.BCELoss()
    loss2 = nn.BCELoss()
    loss3 = nn.BCELoss()
    #Loading the Trainers
    trainer1 = TrainModel(model1, opt1, loss1)
    trainer2 = TrainModel(model2, opt2, loss2)
    trainer3 = TrainModel(model3, opt3, loss3)
    # Training and saving models
    trainer1.train(train_loader, 60, gpu=True)
    trainer2.train(train_loader, 60, gpu=True)
    trainer3.train(train_loader, 60, gpu=True)

    trained_model1 = trainer1.get_model()
    torch.save(trained_model1.state_dict(), "models/inceptionv4_final.pt")
    trained_model2 = trainer2.get_model()
    torch.save(trained_model2.state_dict(), "models/customcnn_final.pt")
    trained_model3 = trainer3.get_model()
    torch.save(trained_model3.state_dict(), "models/alexnet_final.pt")


def split_set(df:pl.DataFrame, train_size:float=0.0, test_size:float=0.0, valid_size:float=0.0):
    """Split the dataset into train, test, and validation sets.

    Splits a polars DataFrame into pieces or resamples the DataFrame
    By using the length of the DataFrame with the train, test, and
    validate set sizes to split the whole DataFrame into three
    proportions.

    Parameters
    ----------
    df : Polars DataFrame
        The entire Dataset that will be split into pieces.
    train_size : Int
        value between 0.00 and 1.00 which will be used to determine
        the train set size.
    test_size : Int
        value between 0.00 and 1.00 which will be used to determine
        the test set size.
    valid_size : Int
        value between 0.00 and 1.00 which will be used to determine
        the validation set size.
    """
    assert sum([train_size, test_size, valid_size]) <= 1.00, "The sum of Train, Test, and Valid sets should be equal to 100%."
    df.sample(fraction=1, shuffle=True, seed=42)
    if sum([train_size, test_size, valid_size]) == 0.0:
        return df
    if train_size == 1.00 or test_size == 1.00 or valid_size == 1.00:
        return df
    elif sum([train_size, test_size]) == 1.00:
        train_length = round(len(df) * train_size)
        train_set = df.slice(0, train_length)
        test_length = round(len(df) * test_size)
        test_offset = train_length + 1
        test_set = df.slice(test_offset, test_length)
        return train_set, test_set
    elif sum([train_size, valid_size]) == 1.00:
        train_length = round(len(df) * train_size)
        train_set = df.slice(0, train_length)
        valid_length = round(len(df) * valid_size)
        valid_offset = train_length + 1
        valid_set = df.slice(valid_offset, valid_length)
        return train_set, valid_set
    elif sum([test_size, valid_size]) == 1.00:
        test_length = round(len(df) * test_size)
        test_set = df.slice(0, test_length)
        valid_length = round(len(df) * valid_size)
        valid_offset = test_length + 1
        valid_set = df.slice(valid_offset, valid_length)
        return test_set, valid_set
    elif sum([train_size, test_size, valid_size]) == 1.00:
        train_length = round(len(df) * train_size)
        train_set = df.slice(0, train_length)
        test_length = round(len(df) * test_size)
        test_offset = train_length + 1
        test_set = df.slice(test_offset, test_length)
        valid_length = round(len(df) * valid_size)
        valid_offset = train_length + test_length + 1
        valid_set = df.slice(valid_offset, valid_length)
        return train_set, test_set, valid_set
    else:
        pass

def change_column_names(df:pl.DataFrame) -> pl.DataFrame:
    """Change the column name by replacing space with underline and decapitalize words.

    Extracts all of the columns, changes all capitalized characters to
    its lowercase version. The new column names are then used to
    replace the old column names within the DataFrame.

    Parameters
    ----------
    df : Polars DataFrame
        The DataFrame containing all of the data.

    Returns
    -------
    Polars DataFrame
        The DataFrame with the column name changes.
    """
    columns = df.columns
    col_changes = dict()
    for col in columns:
        ncol = col.strip()
        ncol = col.lower()
        ncol = col.replace(" ", "_")
        col_changes[col] = ncol
    df = df.rename(col_changes)
    return df

def add_label_from_path(root:str, search_labels:dict[str]) -> pl.DataFrame:
    """Add a label based on terms within paths.

    Loads a csv file and searches for specific terms found as keys
    within the dictionary and labels them based on the value associated
    with the key. This label is added to a column named image type.

    Parameters
    ----------
    root : String
        The file containing the list of paths to be searched on. This
        will be loaded as a polars DataFrame.
    search_labels : Dictionary [String]
        Contains key:value pairs that are meant to be the search term
        and the label respectively.

    Returns
    -------
    Polars DataFrame
        Modified DataFrame containing the labeled paths.
    """
    assert ".csv" in root, TypeError("File is not in CSV format.")
    df = pl.read_csv(root)
    mod_data = list()
    for row in df.iter_rows(named=True):
        for term, label in search_labels.items():
            path = row['path']
            lpath = path.lower()
            lterm = term.lower()
            if lterm in lpath:
                mod_data.append({"path":path, "type":label})
            else:
                pass
    df_new = pl.DataFrame(mod_data)
    df_merged = df.join(df_new,on="path", how="inner")
    return df_merged

def update_version(filename:str, new_model:bool=False):
    """Save the model version and update it.

    Parameters
    ----------
    filename : str
        path to file containing the version.
    """
    if new_model == True:
        with open(filename, 'r+') as fp:
            cv = int(fp.read())
            nv = cv + 1
            fp.seek(0)
            fp.truncate()
            fp.write(str(nv))
            fp.close()
    else:
        with open(filename, 'r+') as fp:
            cv = int(fp.read())
            nv = 1
            fp.seek(0)
            fp.truncate()
            fp.write(str(nv))
            fp.close()

def create_target_transform(n_class:int, val:int=1):
    """Create the transformation function for the label data.

    Uses the pytorch Lambda transform to recreate lable data as a
    vector with the length equal to the number of classes. This will
    allow one to use the transform for any situation and dataset
    class that one may use.

    Parameters
    ----------
    n_class : int
        The number of classes.
    val : int
        The default value for the classes.

    Returns
    -------
    pytorch Transform
        The transform function for the label data.
    """
    target_transform = transforms.Lambda(lambda y: torch.zeros(n_class, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=val))
    return target_transform

def extract_metadata(root:str, cols:list=[]):
    """Extract the metadata from the DICOM FILE.

    Extracts the data from DICOM files based on the list of labels
    written, collects the paths of all files from the directory and
    places the data within a Polars DataFrame.

    Parameters
    ----------
    root : str
        The main directory containing the data

    cols : list
        list of features found within the dicom file.

    Returns
    -------
    Polars DataFrame
        Dataframe containing all of the desired metadata.
    """
    files = get_file_paths(root)
    dset = list()
    for file in files:
        if '.dcm' in file:
            file = file[:-2]
            dcimg = dcmread(file)
            datapoint = { str(col):dcimg[str(col)].value for col in cols }
            datapoint['unique id'] = str(str(dcimg['PatientID'].value) + '-' + str(dcimg['InstanceNumber'].value))
            datapoint['path'] = file
            dset.append(
                    datapoint
                    )
    df = pl.DataFrame(dset)
    return df

def convert_string_to_cat(df:pl.DataFrame, col:str|list) -> pl.DataFrame:
    """Convert the string column to categorical column.

    Uses polars' casting capabilities to transform either one or
    multiple columns from the string datatype to the categorical
    datatype.

    Parameters
    ----------
    df : Polars DataFrame
        The dataframe containing the columns.
    col : String or List
        The variable containing the column(s).
    Returns
    -------
    Polars DataFrame
        Contains the new dataframe with the converted columns.

    Examples
    --------
    >>> from pipeline import convert_string_to_cat
    >>> import polars as pl
    >>> df = pl.DataFrame({
    ...     "cat": ['hello', 'goodbye'],
    ...     "bar": [1, 2]
    ... })
    >>> df = convert_string_to_cat(df, 'cat')

    Now that we have the column within the correct type one can
    make the categorical data into a numerical type through the
    following code.

    >>> df = df.with_columns(pl.col('cat').to_physical)

    """
    if type(col) == str:
        df = df.with_columns(pl.col(col).cast(pl.Categorical))
    elif type(col) == list:
        df = df.with_columns(list(pl.col(c).cast(pl.Categorical) for c in col))
    else:
        print("Cannot process type: {}".format(type(col)))
        exit()
    return df

def get_file_paths(root:str, filename:str=None) -> list[str]:
    """Get the path to all files within a folder.

    Search through a root directory to extract the path of all files
    and extract the file path of the file regardless of the depth of
    the file within the directory.

    Parameters
    ----------
    root : String
        The root to the file directory.
    filename : String
        Name of the file containing the list of files within the
        root path. Stated to be None if not desired.

    Returns
    -------
    List
        Contains the full (relative) path to the files within the
        root directory.
    """
    all_files = list()
    for path, subdirs, files in os.walk(root):
        for name in files:
            all_files.append(os.path.join(path, name, '\n'))
    if filename != None:
        with open(filename, 'w') as fp:
            fp.writelines(all_files)
            fp.close()
    return all_files

def gather_segmentation_images(filename:str, paths:str, id:str):
    """Get all of the Images with Segmentations.

    Gathers all of the image slices together with the
    respective segmentations. As this only uses the Patient
    ID as the unique identifier, only one of the folders
    after the patient id directory will be chosen together
    with the image slices. The most consistent folder may
    be used as all patients will share this folder.

    Parameters
    ----------
    filename : string
        filename containing the training data set with the
        bounding boxes, and the slices or range of slices.

    paths : string
        text file containing all of the paths to the image
        files or slices.
    id : string
        the unique identifier for the sample.
    """
    df = pl.read_csv(filename)
    with open(paths, 'r') as fp:
        list__paths = fp.readlines()
        fp.close()
    for _, row in df.iter_rows():
        patient_folder = list(filter(lambda x: row[id] in x, list__paths))
        print(patient_folder)
        exit()

def load_image(filename:str, size:tuple|int) -> np.ndarray:
    """Load the image based on the path.

    Parameters
    ----------
    filename : string
        string containing the relative or absolute path to
        the image.
    size : tuple | int
        tuple containing the desired width and height to
        readjust the image. In the case that the image is
        square, then the size may be an integer.

    Returns
    -------
    numpy Array
        Returns a 3D array containing the image of the
        dimensions (width, height, colors).

    """
    img = Image.open( filename ).convert('L')
    if type(size) == int:
        tsize = (size, size)
        img = img.resize(tsize)
    else:
        img = img.resize(size)
    img.load()
    if 'mask' in filename:
        data = np.asarray( img ).astype('int32')
    else:
        raw_data = np.asarray( img ).astype('float32')
        data = (raw_data - np.min(raw_data)) / (np.max(raw_data) - np.min(raw_data))
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    else:
        pass
    return data

def merge_dictionaries(*dictionaries) -> dict:
    """Merge n number of dictionaries.

    Parameters
    ----------
    dictionaries : list of dictionaries
        Contains dictionaries with related data. These dictionaries
        represent a separate data point each.

    Returns
    -------
    dictionary
        Merged dictionary containing lists associated with their own
        keys. These keys refer to columns and the lists refer to
        the values associated with the key.

    """
    mdictionary = defaultdict()
    for dictionary in dictionaries:
        for key, value in dictionary.items():
            if key not in mdictionary:
                mdictionary[key] = [value]
            else:
                mdictionary[key].append(value)
    return mdictionary

def balance_data(df:pl.DataFrame, columns:list=[],sample_size:int=None) -> pl.DataFrame:
    """Balance data for model training.

    Splits the dataset into groups based on the categorical
    columns provided. The function will use a for loop to
    extract samples based on predetermined categories. a
    list of permutations will be used.

    Parameters
    ----------
    df : Polars DataFrame
        Contains all of the data necessary to load the
        training data set.
    columns : list
        List of columns which will be used to categorize
        the data. In the case that the columns list is
        empty, then the dataset will simply be resampled.
    sample_size : integer
        Describes the sample size of the dataset that
        will be used for either training or testing the
        machine learning model.

    Returns
    -------
    Polars DataFrame
        Balanced data set ready for feature extraction.

    """
    assert sample_size != 0, "The sample size cannot be zero."
    if sample_size == None:
        sample_size = len(df)
    else:
        pass

    if columns == []:
        df_balanced = df.sample(n=sample_size, seed=42)
    else:
        groups = df.group_by(columns)
        df.filter(
                pl.int_range(0, pl.count()).shuffle().over(columns) <= (sample_size / len(groups.count()))
                )
        df_balanced = df
        #groups = df.groupby(columns)
        #number_groups = len(groups.groups)
        #sample_group_size = int(sample_size / number_groups)
        #sampled_groups = list()
        #diff_sample_size = 0
        #for gtype, df_group in groups:
        #    fgroup = sample_group_size + diff_sample_size
        #    if len(df_group) >= fgroup:
        #        df__selected_group = df_group.sample(n=int(fgroup), random_state=42)
        #    elif len(df_group) >= sample_group_size:
        #        df__selected_group = df_group.sample(n=int(sample_group_size), random_state=42)
        #    elif fgroup <= 0:
        #        break
        #    else:
        #        df__selected_group = df_group.sample(n=int(len(df_group)), random_state=42)
        #    sampled_groups.append(df__selected_group)
        #    diff_sample_size += sample_group_size - len(df__selected_group)
        #df_balanced = pl.concat(sampled_groups)
    return df_balanced

def rescale_image(img:np.ndarray) -> np.ndarray:
    """Rescale the image to a more manageable size.

    Changes the size of the image based on the length and
    width of the image itself. This is to reduce the amount
    of computations required to make predictions based on
    the image.

    Parameters
    ----------
    img : Numpy Array
        array containing the raw values of images.

    Returns
    -------
    Numpy Array
        Array containing the rescaled image.

    """
    size = img.shape
    width = int(size[1] / 2)
    height = int(size[0] / 2)
    img = img.astype(float)
    scaled_image = (np.maximum(img, 0) / img.max()) * 255
    scaled_image = np.uint8(scaled_image)
    final_image = Image.fromarray(scaled_image)
    final_image = final_image.resize(size=(width, height))
    img_mod = np.asarray(final_image)
    img_mod = np.asarray([img_mod])
    img_mod = np.moveaxis(img_mod, 0, -1)
    return img_mod

def calculate_confusion_matrix(fin_predictions:pl.DataFrame):
    """Calculate the confusion matrix using pandas.

    Calculates the confusion matrix using a csv file that
    contains both the predictions and actual labels. This
    function then creates a crosstab of the data to develop
    the confusion matrix.

    Parameters
    ----------
    fin_predictions : Pandas DataFrame
        DataFrame containing the prediction and actual
        labels.

    Returns
    -------
    Polars DataFrame
        Cross tab containing the confusion matrix of the
        predictions compared to the actual labels.

    Dictionary
        Contains the basic metrics obtained from the
        confusion matrix. The metrics are the following:
        - Accuracy
        - Precision
        - Recall
        - F1 Score
    """
    ct = fin_predictions.pivot(values="predictions", index="classification", columns="classification", aggregate_function='count')
    # Set the initial values
    tp = ct.values[1][1]
    tn = ct.values[0][0]
    fn = ct.values[0][1]
    fp = ct.values[1][0]
    # Calculate the metrics
    metrics = dict()
    metrics['Accuracy'] = (tp + tn) / (tp + tn + fp + fn) # Ability of model to get the correct predictions
    metrics['Precision'] = tp / (tp + fp) # Ability of model to label actual positives as positives (think retrospectively)
    metrics['Recall'] = tp / (tp + fn) # Ability of model to correctly identify positives
    metrics['F1 Score'] = (2 * metrics['Precision'] * metrics['Recall']) / (metrics['Precision'] + metrics['Recall'])
    return ct, metrics

activation={}
def get_activation(name):
    """Extract the activation of a specific layer."""
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


if __name__ == "__main__":
    _main()
