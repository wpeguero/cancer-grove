"""Custom Dataset Classes that Inherit from Pytorch's Dataset Class."""

import os

from torch.utils import data
import polars as pl
from pydicom import dcmread
import numpy as np
import torch
from PIL import Image


def _main():
    """Test the new functions."""
    pass


class ImageSet(data.Dataset):
    """Dataset that will load unlabeled images.

    *Alternatively, one can use the torchvision.data.ImageFolder class for the same reason.*

    Parameters
    ----------
    root : str
        Path to the folder containing the data.
    """

    def __init__(self, root="train/", image_loader=None, transform=None):
        """Initialize the Dataset Subclass."""
        self.root = root
        self.folders = os.listdir(root)
        self.files = list()
        self.dict__files = dict()
        for folder in self.folders:
            fold = os.path.join(self.root, folder)
            self.dict__files[folder] = os.listdir(fold)
            self.files.extend(os.listdir(fold))
        self.loader = image_loader
        self.transform = transform

    def __len__(self):
        """Get the Length of the items within the dataset."""
        return sum([len(self.files)])

    def __getitem__(self, index):
        """Get item from class."""
        images = [
            self.loader(os.path.join(self.root, folder)) for folder in self.folders
        ]
        if self.transform is not None:
            images = [self.transform(img) for img in images]
        return images


class MixedDataset(data.Dataset):
    """Dataset that inputs image & categorical data.

    Parameters
    ----------
    root : str
        directory containing all of the images.
    csvfile : str | Polars DataFrame
        path to the csv with the categorical data or the loaded file
        using the Polars library.
    label_column : str
        Column containing the label about cancer.
    """

    def __init__(
        self,
        csvfile: str | pl.DataFrame,
        label_column: str = "pathology",
        image_loader=None,
        image_transforms=None,
        cat_transforms=None,
    ):
        """Initialize the class."""
        if isinstance(csvfile, str):
            self.csv = pl.read_csv(csvfile)
        else:
            self.csv = csvfile
        self.lcol = label_column
        self.loader = image_loader
        self.image_transforms = image_transforms
        self.cat_transforms = cat_transforms

    def __len__(self):
        """Calculate the length of the dataset."""
        return self.csv.select(pl.count()).item()

    def __getitem__(self, index):
        """Get the datapoint."""
        if torch.is_tensor(index):
            index.tolist()
        image = Image.open(self.csv["path"][index])
        if self.image_transforms:
            image = self.image_transforms(image)
        supplementary_data = np.array(self.csv.select(pl.exclude("path", self.lcol)))
        labels = np.array(self.csv.select(self.lcol))
        if self.cat_transforms:
            labels = self.label_transforms(labels)
        sample = {
            "image": image,
            "supplementary data": supplementary_data,
            "labels": labels,
        }
        return sample


class DICOMSet(data.Dataset):
    """Dataset used to load and extract information from DICOM images.

    Loads image from the dicom files and adds a label associated with
    said image path.

    Parameters
    ----------
    csvfile : String or Polars DataFrame
        File or path to file containing the path to the image and the
        categorical data.
    label_col : String
        The column containing the labels for the classifier.
    img_col : String
        The column containing the path to the dicom file.
    """

    def __init__(
        self,
        csvfile: str | pl.DataFrame,
        label_col: str,
        img_col: str = "paths",
        image_loader=None,
        image_transforms=None,
        categorical_transforms=None,
    ):
        """Init the Class."""
        assert isinstance(csvfile, str) or isinstance(csvfile, pl.DataFrame), TypeError(
            "csvfile is not of the correct type, the current type is {}".format(
                type(csvfile)
            )
        )
        if isinstance(csvfile, str):
            self.csv = pl.read_csv(csvfile)
        else:
            self.csv = csvfile
        self.lcol = label_col
        self.pcol = img_col
        self.loader = image_loader
        self.img_transforms = image_transforms
        self.cat_transforms = categorical_transforms

    def __len__(self):
        """Calculate the length of the dataset."""
        return len(self.csv)
        # return self.csv.select(pl.count()).item()

    def __getitem__(self, index):
        """Get the datapoint."""
        if torch.is_tensor(index):
            index.tolist()
        dicom_file = dcmread(self.csv.select(self.pcol).row(index)[0])
        img = self.extract_image(dicom_file)
        cat = self.csv.select(pl.col(self.lcol).cast(pl.Int8)).row(index)[0]
        if self.img_transforms:
            img = self.img_transforms(img)
        if self.cat_transforms:
            cat = self.cat_transforms(cat)
        # sample = {'image': img, 'labels': cat}
        return img, cat

    @staticmethod
    def extract_image(dicom_file):
        """Extract image from the DICOM File."""
        slices = np.asarray(dicom_file.pixel_array).astype("float32")
        if slices.ndim <= 2:
            slice = slices
        elif slices.ndim > 3:
            slice = slices[0]
        slice = slice[..., np.newaxis]
        return slice

    @staticmethod
    def extract_metadata(dicom_file, cols: list[str]) -> dict:
        """Extract metadata from the DICOM file.

        Parameters
        ----------
        dicom_file
            DICOM file containing the desired metadata and image.

        cols : list
            labels of the metadata contained within the DICOM file.

        Returns
        -------
        Dictionary
            Contains the label from the columns and the associated
            value.
        """
        return {str(col): dicom_file[str(col)].value for col in cols}


if __name__ == "__main__":
    _main()
