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
from trainers import TrainModel, VERSION
import models

img_size = (512, 512)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

def _main():
    """Test the new functions."""
    fn__images = "data/Chest_CT_Scans/train/"
    fn__test_images = "data/Chest_CT_Scans/test/"
    classes = ('adenocarcinoma', 'large.cell.carcinoma', 'normal', 'squamous.cell.carcinoma')
    #fn__images = "data/cat_loaf_set/"
    img_transforms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    target_transform = transforms.Lambda(lambda y: torch.zeros(4, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
    dset = datasets.ImageFolder(fn__images, transform=img_transforms, target_transform=target_transform)
    testset = datasets.ImageFolder(fn__test_images, transform=img_transforms, target_transform=target_transform)
    dloader = data.DataLoader(dset, shuffle=True, batch_size=16, num_workers=4)
    testloader = data.DataLoader(testset, shuffle=True, batch_size=16, num_workers=4)
    #model = TutorialNet(3, 4)
    #model = CustomCNN(3, 4)
    #model = AlexNet(3, 4)
    model = InceptionV4(4)
    #model = nn.Sequential(
    #        InceptionStem(3),
    #        InceptionA(384),
    #        ReductionA(384),
    #        InceptionB(1024),
    #        ReductionB(1024),
    #        InceptionC(1536),
    #        nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
    #        nn.Flatten(),
    #        nn.Dropout(0.8),
    #        nn.Softmax(4)
    #        )
    opt = optim.SGD(model.parameters(), lr=0.003, weight_decay=0.005, momentum=0.9)
    loss = nn.CrossEntropyLoss()
    # Sample image for the sake of testing
    #img = Image.open("data/Chest_CT_Scans/test/squamous.cell.carcinoma/000129 (6).png")
    #print(img)
    #datapoint = np.asarray([img, np.array([1, 2, 3, 4])])
    #model(img.unsqueeze(0))
    trainer = TrainModel(model, opt, loss)
    trainer.train(dloader, 100, gpu=True)
    trainer.test(testloader, classes, gpu=True, version=VERSION)
    model = trainer.get_model()
    torch.save(model.state_dict(), 'models/{}_model_{}.pt'.format(model.__class__.__name__, VERSION))
    with open('src/model_version.txt', 'r+') as fp:
        cv = int(fp.read())
        nv = cv + 1
        fp.seek(0)
        fp.truncate()
        fp.write(str(nv))
        fp.close()
    #img = img_transforms(img)
    #model.register_forward_hook(get_activation('conv1'))
    #with torch.no_grad():
    #    output = model.conv1(img.to('cuda').unsqueeze(0))
    #fig = px.imshow(output.to('cpu')[0], facet_col=0, facet_col_wrap=5)
    #fig.show()


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

def _get_list_of_files(root:str) -> pl.DataFrame:
    """Get the list of files within given directory."""
    p = pathlib.Path(root).glob('**/*')
    files = [x for x in p if x.is_file()]
    data = list()
    for file in files:
        parts = str(file).split('/')
        data.append({'dirname':parts[-2], 'path':str(file)})
    df = pl.DataFrame(data)
    return df

def _rename_folders(filepath:str):
    df__dicom_info = pl.read_csv(fn__dicom_info)
    root_path = 'data/CBIS/jpeg/'
    list__folders = list() # Contains the new path of the renamed folders that contain images.
    for row in df__dicom_info.iter_rows(named=True):
        original_path = str(row['image_path']).split('/')
        folder_path = os.path.join(root_path, original_path[2])
        new_path = f'{root_path}{row["PatientID"]}'
        if (os.path.exists(new_path)) and (os.path.exists(folder_path)):
            files = os.listdir(folder_path)
            for file in files:
                os.replace(os.path.join(folder_path, file), os.path.join(new_path, file))
        elif (os.path.exists(folder_path) == True) and (os.path.exists(new_path) == False):
            os.rename(folder_path, new_path)
        else:
            pass
        list__folders.append({'PatientID':row['PatientID'], 'new_path':os.path.join(root_path, row['PatientID'])})
    df__paths = pl.DataFrame(list__folders)
    df__paths.write_csv('data/CBIS/csv/image_paths.csv')

def _extract_feature_definitions(filepath:str, savepath:str, l:int):
    df = pl.read_csv(filepath)
    features = df.iloc[:l]
    feats = features.fillna("blank")
    with open(savepath, 'w') as fp:
        json.dump(feats, fp)
        fp.close()

def _remove_first_row(filepath:str, nfilepath:str):
    xls = pl.ExcelFile(filepath, engine='xlrd')
    df = pl.read_excel(xls, 0)
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
    df = pl.read_csv(filename)
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
    return None

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

def extract_key_images(data_dir:str, metadata_filename:str, new_download = False):
    """Extract the key images based on the Annotation Boxes file.

    Grabs the images from the full directory and
    moves them to a separate directory for keeping
    only the key data.

    Parameters
    ----------
    dat_dir : str
        The path to the images.
    metadata_filename : str
        name of the file containing superfluous data about the image
        (i.e. the age of the patient, the image orientation, etc.).
    new_download : bool
        Value used to determine whether the paths have already been
        extracted.

    Returns
    -------
    None
        This is returned in the case that the download is not new.

    or

    Polars.DataFrame
        Contains the image paths with the metadata associated with
        the image.
    """
    if not new_download:
        return None
    else:
        df__metadata = pl.read_csv(metadata_filename)
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
        df_img_paths = pl.DataFrame(img_paths_list)
        return df_img_paths

def extract_dicom_data(file, target_data:list =[]) -> dict:
    """Extract the data from the .dcm files.

    Reads each independent file using the pydicom
    library and extracts key information, such as
    the age, sex, ethnicity, weight of the patient,
    and the imaging modality used.

    Parameters
    ---------
    file : str or pydicom.Dataset
        Either the path to the file or pydicom Dataset.
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
    dictionary
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
    datapoint = dict()
    if type(file) == str:
        try:
            ds = dcmread(file)
            datapoint['Full Location'] = file
        except (InvalidDicomError) as e:
            print(f"ERROR: The file {file} is not a DICOM file and therefore cannot be read.")
            print(e)
            exit()
    else:
        ds = file

    slices = np.asarray(ds.pixel_array).astype('float32')
    #slices = da.asarray(ds.pixel_array).astype('float32')
    #slices = (slices - np.min(slices)) / (np.max(slices) - np.min(slices))
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
    slices = slices[..., np.newaxis]
    datapoint['image'] = slices
    datapoint['Patient ID'] = ds.PatientID
    return datapoint

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

def transform_dicom_data(datapoint:dict, definitions:dict) -> dict:
    """Transform the data into an format that can be used for displaying and modeling.

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
        the meaning of each categorical value.
    Returns
    -------
    dictionary
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
    except (AttributeError, KeyError):
        print('WARNING: Indicator "image" does not exist.')
    return datapoint

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

def load_training_data(filename:str, pathcol:str, balance:bool=True, sample_size:int=1_000, cat_labels:list=[]):
    """Load the DICOM data as a dictionary.

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

    cat_labels : list
        Contains all of the labels that will be used within
        the training set. These labels are meant to be the
        column names of the categorical values that will be
        used for training the machine learning model.
    Returns
    -------
    dictionary
        Dictionary containing the encoded values
        for the metadata and the transformed image
        for input to the model.
    """
    if type(filename) == str:
        df = pl.read_csv(filename)
    elif type(filename) == pl.DataFrame:
        df = filename
    else:
        print("There was some error.")
        exit()
    #data = dict()
    if balance == True:
        df_balanced = balance_data(df, sample_size=sample_size)
    else:
        df_balanced = df.sample(n=sample_size, seed=42)

    if bool(cat_labels) == False:
        data = map(extract_data, df_balanced[pathcol])
        df = pl.DataFrame(list(data))
        df_full = df_balanced.join(df, on=pathcol)
        return df_full
    elif bool(cat_labels) == True:
        full_labels = cat_labels * len(cat_labels) * len(df_balanced)
        data = map(extract_data, df_balanced[pathcol], full_labels)
        df = pl.DataFrame(list(data))
        df_full = df.join(df_balanced, on=pathcol)
        return df_full
    else:
        print('None of the conditions were met')
        exit()

def  load_testing_data(filename:str, sample_size= 1_000) -> pl.DataFrame:
    """Load the data used  for testing.

    Loads a dataset to be fed into the model for making
    predictions. The output of the testing data will be
    comprised of a dictionary that can be fed directly into
    the model.

    Parameters
    ----------
    filename : str
        path to file containing the file paths to test data.

    Returns
    -------
    Polars DataFrame
        Contains the all of the data necessary for testing.
    """
    df = pl.read_csv(filename)
    df = df.dropna(subset=['classification'])
    df = df.sample(n=sample_size, seed=42)
    print("iterating through {} rows...".format(len(df)))
    dfp_list = list()
    for _, row in df.iterrows():
        datapoint = extract_data(row['paths'])
        datapoint = transform_data(datapoint)
        drow = row.to_dict()
        datapoint.update(drow)
        dfp_list.append(datapoint)
    tdata = pl.DataFrame(dfp_list)
    return tdata

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
