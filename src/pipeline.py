"""Pipeline Module
------------------

Algorithms used to process data before modeling.

...

A set of algorithms used to feed in and process
data before used within the model. This will contain
the data extraction from its rawest form and output
the final form of the data set. The main source of
data will be image related from the Cancer Imaging
Archive.
"""
import os
import pathlib
from fractions import Fraction
import json

import numpy as np
import pandas as pd
from pydicom import dcmread
import pydicom as pdcm
from PIL import Image
import matplotlib.pyplot as plt
from pydicom.errors import InvalidDicomError
from keras.layers import CategoryEncoding
from keras.models import load_model
from tensorflow.nn import softmax
import plotly.express as px


##The dataset had duplicates due to images without any data provided on the clinical analysis. Some images were taken without clinical data for the purpose of simply taking the image. Nothing was identified for these and therefore these should be removed from  the dataset before converting the .dcm files into .png files.
def _main():
    """Test the new functions."""
    dicom__file = "data/DBCMRI/Duke-Breast-Cancer-MRI/Breast_MRI_002/01-01-1990-NA-MRI BREAST BILATERAL W  WO-51972/600.000000-ax 3d dyn-25442/1-125.dcm"
    extract_data(dicom__file)


def _extract_feature_definitions(filepath:str, savepath:str, l:int):
    df = pd.read_csv(filepath)
    features = df.iloc[:l]
    feats = features.fillna("blank")
    with open(savepath, 'w') as fp:
        json.dump(feats, fp)
        fp.close()

def _remove_first_row(filepath:str, nfilepath:str):
    xls = pd.ExcelFile(filepath, engine='xlrd')
    df = pd.read_excel(xls, 0)
    df.to_csv(filepath, index=False)
    with open(filepath, 'r') as file:
        data = file.read()
    new_data = data.split('\n', 1)[-1]
    with open(nfilepath, 'w') as fp:
        fp.write(new_data)

def _convert_dicom_to_png(filename:str) -> None:
    """Convert a list of dicom files into their png forms.
    
    ...
    """
    df = pd.read_csv(filename)
    for _, row in df.iterrows():
        ds = dcmread(row['paths'])
        path = pathlib.PurePath(row['paths'])
        dicom_name = path.name
        name = dicom_name.replace(".dcm", "")
        new_image = ds.pixel_array.astype(float)
        scaled_image = (np.maximum(new_image, 0) / new_image.max()) * 255
        scaled_image = np.uint8(scaled_image)
        final_image = Image.fromarray(scaled_image)
        final_image.save(f"data/CMMD-set/classifying_set/raw_png/{row['Subject ID'] + '_' + name + ds.ImageLaterality}.png")

def _extract_key_images(data_dir:str, metadata_filename:str, new_download = False):
    """Extract the key images based on the Annotation Boxes file.
    
    ...

    Grabs the images from the full directory and
    moves them to a separate directory for keeping
    only the key data."""
    if not new_download:
        return None
    else:
        df__metadata = pd.read_csv(metadata_filename)
        root_path = os.getcwd()
        root_path = root_path.replace("//", "/")
        img_paths_list = list()
        for _, row in df__metadata.iterrows():
            PID = row["Subject ID"]
            file_location = row["File Location"]
            file_location = file_location.replace("//","/").lstrip(".")
            file_location = root_path + data_dir + file_location
            imgs = os.listdir(file_location)
            for img in imgs:
                ds = dcmread(file_location + '/' + img)
                img_paths = {
                    'ID1': PID,
                    'paths': file_location + '/' + img,
                    'LeftRight': ds.ImageLaterality
                }
                img_paths_list.append(img_paths)
        df_img_paths = pd.DataFrame(img_paths_list)
        return df_img_paths

def extract_data(file, target_data:list =[]) -> dict:
    """Extract the data from the .dcm files.

    ...

    Reads each independent file using the pydicom
    library and extracts key information, such as
    the age, sex, ethnicity, weight of the patient,
    and the imaging modality used.

    Parameters
    ---------
    file : Unknown 
        Either the path to the file or the file itself.
        In the case that the .dcm file is already
        loaded, the algorithm will proceed to extract
        the data. Otherwise, the algorithm will load
        the .dcm file and extract the necessary data.
    
    target_data : List
        This contains all of the tag names that will be
        used as part of the data extraction. In the case
        that the list is empty, then only the image will be
        used.

    Returns
    -------
    datapoint : dictionary
        Dictionary comprised of the image data
        (numpy array), and the metadata associated
        with the DICOM file as its own separate
        `key:value` pair. This only pertains to the
        patient data and NOT the metadata describing
        how the image was taken.
    
    Raises
    ------
    InvalidDicomError
        The file selected for reading is not a DICOM
        or does not end in .dcm. Set in place to
        stop the algorithm in the case that any other
        filetype is introduced. Causes an error to be
        printed and the program to exit.
    
    AttributeError
        Occurs in the case that the DICOM file does
        not contain some of the metadata used for
        classifying the patient. In the case that
        the metadata does not exist, then the model
        continues on with the classification and some
        plots may be missing from the second page.
    """
    if type(file) == str:
        try:
            ds = dcmread(file)
        except (InvalidDicomError) as e:
            print(f"ERROR: The file {file} is not a DICOM file and therefore cannot be read.")
            exit()
    else:
        ds = file
    datapoint = dict()
    slices = ds.pixel_array
    if target_data == []:
        pass
    else:
        for target in target_data:
            if target in ds:
                datapoint[str(target)] = ds[target].value
            else:
                pass
    if slices.ndim <= 2:
        pass
    elif slices.ndim >= 3:
        slices = slices[0]
    datapoint['image'] = slices
    datapoint['Patient ID'] = ds.PatientID
    return datapoint

def transform_data(datapoint:dict, definitions:dict) -> dict:
    """ Transform the data into an format that can be used for displaying and modeling.

    ...

    Transforms the textual categorical data into numerical
    to input the data into the machine learning model. This
    function depends upon two dictionaries, one containing
    the data and the other a set of references that can be
    used to transform the textual categorical values into
    the numerical values. This function also removes the 
    area of the image that contains columns whose values
    are zero.

    Parameters
    ----------
    datapoint : dictionary
        Contains the image and related metadata in
        `key:value` pair format.
    
    definitions : dictionary
        Set of values found within the data point and their
        definitions. This will contain the column value and
        the meaning of each categorical value. The nature
        of this could be the following:
        EX.: {
            key:{
                "category":1
                }
            }
    
    Returns
    -------
    datapoint :dictionary
        same dictionary with the categorical data
        transformed into numerical (from text).
    
    Raises
    ------
    AttributeError
        Indicator of the `key` does not exists.
    
    KeyError
        Indicator of the `key` does not exists.
    """
    for key, values in definitions.items():
        if key in datapoint.keys():
            datapoint[key] = values[datapoint.get(key)]
        else:
            print(f'WARNING: Indicator "{key}" could not be found within the data point.')
    try:
        img = datapoint['image']
        img = img[:, ~np.all(img == 0, axis = 0)]
        img_mod = rescale_image(img)
        datapoint['image'] = img_mod
    except (AttributeError, KeyError) as e:
        print('WARNING: Indicator "image" does not exist.')
    return datapoint

def balance_data(df:pd.DataFrame, columns:list=[],sample_size:int=1000) -> pd.DataFrame:
    """Balance data for model training.
    
    Splits the dataset into groups based on the categorical
    columns provided. The function will use a for loop to
    extract samples based on predetermined categories. a
    list of permutations will be used 

    Parameter(s)
    ------------
    df : Pandas DataFrame
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
    df_balanced : Pandas DataFrame
        Balanced data set ready for feature extraction.
    """
    if columns == []:
        df_balanced = df.sample(frac=1, random_state=42)
    else:
        groups = df.groupby(columns)
        igroups = len(groups.groups)
        sgroup = int(sample_size / igroups)
        dgroups = list()
        dsample_size = 0
        for gtype, df_group in groups:
            fgroup = sgroup + dsample_size
            if len(df_group) >= fgroup:
                df__selected_group = df_group.sample(n=int(fgroup), random_state=42)
            elif len(df_group) >= sgroup:
                df__selected_group = df_group.sample(n=int(sgroup), random_state=42)
            else:
                df__selected_group = df_group.sample(n=int(len(df_group)), random_state=42)
            dgroups.append(df__selected_group)
            dsample_size += sgroup - len(df_group)
        df_balanced = pd.concat(dgroups)
    return df_balanced

def load_training_data(filename:str, pathcol:str, validate:bool=False, ssize:int=1000, cat_labels:list=[]):
    """Load the DICOM data as a dictionary.
    ...

    Creates a dictionary containing three different
    numpy arrays. The first array is comprised of
    multiple DICOM images, the second contains the
    categorical data as a vector, and the third contains
    the classification in numerical form.

    Parameters
    ----------
    filename : String
        path to a file which contains the metadata,
        classification, and path to the DICOM file.
        Will also contain some sort of ID to better
        identify the samples.
    
    validate : Boolean
        Conditional statement that determines whether the
        data requires a split between training and
        validation. In the case that this is False, then
        the data set is not split between training and
        validation.
    
    cat_labels : unknown
        Contains all of the labels that will be used within
        the training set. These labels are meant to be the
        column names of the categorical values that will be
        used for training the machine learning model.
    Returns
    -------
    data : dictionary
        Dictionary containing the encoded values
        for the metadata and the transformed image
        for input to the model.
    """
    df = pd.read_csv(filename)
    #data = dict()
    if (cat_labels == False and validate == True):
        df_balanced = balance_data(df, sample_size=ssize)
        df_test = df.drop(df_balanced.index)
        df_validate = balance_data(df_test, sample_size=int(0.5*ssize))
        data = list(map(extract_data,df_balanced[pathcol]))
        data_test = list(map(extract_data, df_test[pathcol]))
        data_validate = list(map(extract_data, df_validate[pathcol]))
        df_train = pd.DataFrame(data)
        df_test = pd.DataFrame(df_test)
        df_val = pd.DataFrame(data_validate)
        return df_train, df_test, df_val
    elif (cat_labels == False and validate == False):
        df_balanced = balance_data(df, sample_size=ssize)
        df_test = df.drop(df_balanced.index)
        data = list(map(extract_data, df_balanced[pathcol]))
        data_test = list(map(extract_data, df_test[pathcol]))
        df_train = pd.DataFrame(data)
        df_test = pd.DataFrame(data_test)
        return df_train, df_test
    elif (cat_labels == True and validate == True):
        df_balanced = balance_data(df, sample_size=ssize)
        df_test = df.drop(df_balanced.index)
        df_validate = balance_data(df_test, sample_size=int(0.5*ssize))
        data = list(map(extract_data,df_balanced[pathcol], cat_labels))
        data_test = list(map(extract_data, df_test[pathcol], cat_labels))
        data_validate = list(map(extract_data, df_validate[pathcol], cat_labels))
        df_train = pd.DataFrame(data)
        df_test = pd.DataFrame(df_test)
        df_val = pd.DataFrame(data_validate)
        return df_train, df_test, df_val
    elif (cat_labels == True and validate == False):
        df_balanced = balance_data(df, sample_size=ssize)
        df_test = df.drop(df_balanced.index)
        data = list(map(extract_data, df_balanced[pathcol], pathcol))
        data_test = list(map(extract_data, df_test[pathcol], pathcol))
        df_train = pd.DataFrame(data)
        df_test = pd.DataFrame(data_test)
        return df_train, df_test
    else:
        print('None of the conditions were met')
        exit()

def  load_testing_data(filename:str, sample_size= 1_000) -> pd.DataFrame:
    """Load the data used  for testing.
    
    Loads a dataset to be fed into the model for making
    predictions. The output of the testing data will be
    comprised of a dictionary that can be fed directly into
    the model.

    Parameter(s)
    ------------
    filename : str
        path to file containing the file paths to test data.
    
    Returns
    -------
    df__test : Pandas DataFrame
        Contains the all of the data necessary for testing.
    """
    df = pd.read_csv(filename)
    df = df.dropna(subset=['classification'])
    df = df.sample(n=sample_size, random_state=42)
    print("iterating through {} rows...".format(len(df)))
    dfp_list = list()
    for _, row in df.iterrows():
        datapoint = extract_data(row['paths'])
        datapoint = transform_data(datapoint)
        drow = row.to_dict()
        datapoint.update(drow)
        dfp_list.append(datapoint)
    tdata = pd.DataFrame(dfp_list)
    return tdata

def predict(data:pd.DataFrame, model_name) -> pd.DataFrame:
    """Make predictions based on data provided.

    Extracts the image data using the path column provided
    by the DataFrame argument and uses the model provided
    to make the predictions. The algorithm also extracts
    the necessary categorical data to make the predictions.

    Parameter(s)
    ------------
    data : Pandas DataFrame
        file or object containing the data necessary to
        make predictions. This must contain the path column
        and the categorical columns related to the model.
    
    model_name : str or TensorFlow Model
        either the path to a TensorFlow model or the model
        itself. Used to make predictions on the data.
    
    Returns
    -------
    data : Pandas DataFrame
        predictions together with all of the original
        information.
    """
    if type(model_name) ==str:
        model = load_model(model_name)
    else:
        model = model_name
    fdata = {'image': np.asarray(data['image'].to_list()), 'cat': np.asarray(data[['age', 'side']])}
    predictions = model.predict(fdata, batch_size=5)
    data['sex'] = data['sex'].map(sex)
    data['modality'] = data['modality'].map(modalities)
    data['side'] = data['side'].map(sides)
    if len(predictions) == 1:
        predictions = predictions[0]
        data['score'] = [softmax(predictions).numpy().tolist()]
        data['pred_class'] = class_names[np.argmax(data['score'])]
    elif len(predictions) >= 2:
        pred_data = list()
        for pred in predictions:
            score = softmax(pred)
            pclass = class_names[np.argmax(score)]
            pred_data.append({'score':score.numpy(), 'pred_class':pclass})
        _df = pd.DataFrame(pred_data)
        data = data.join(_df)
    data = data.drop(columns=['image'])
    return data

def rescale_image(img:np.ndarray) -> np.ndarray:
    """Rescale the image to a more manageable size.
    
    Changes the size of the image based on the length and
    width of the image itself. This is to reduce the amount
    of computations required to make predictions based on
    the image.
    
    Parameter(s)
    ------------
    img : Numpy Array
        array containing the raw values of images.
    """
    size = img.shape
    frac = Fraction(size[1], size[0])
    width = frac.numerator
    height = frac.denominator
    img = img.astype(float)
    scaled_image = (np.maximum(img, 0) / img.max()) * 255
    scaled_image = np.uint8(scaled_image)
    final_image = Image.fromarray(scaled_image)
    final_image = final_image.resize(size=(width, height))
    img_mod = np.asarray(final_image)
    img_mod = np.asarray([img_mod])
    img_mod = np.moveaxis(img_mod, 0, -1)
    return img_mod

def calculate_confusion_matrix(fin_predictions:pd.DataFrame):
    """Calculate the confusion matrix using pandas.
    
    Calculates the confusion matrix using a csv file that
    contains both the predictions and actual labels. This
    function then creates a crosstab of the data to develop
    the confusion matrix.
    
    Parameter(s)
    ------------
    fin_predictions : Pandas DataFrame
        DataFrame containing the prediction and actual
        labels.
    
    Returns
    -------
    ct : Pandas DataFrame
        Cross tab containing the confusion matrix of the
        predictions compared to the actual labels.
    
    metrics : Dictionary
        Contains the basic metrics obtained from the
        confusion matrix. The metrics are the following:
        - Accuracy
        - Precision
        - Recall
        - F1 Score
    """
    ct = pd.crosstab(fin_predictions['pred_class'], fin_predictions['classification'])
    print(ct)
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


if __name__ == "__main__":
    _main()
