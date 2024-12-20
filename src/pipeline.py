"""Algorithms used to process data before modeling.

A set of algorithms used to feed in and process
data before used within the model. This will contain
the data extraction from its rawest form and output
the final form of the data set. The main source of
data will be image related from the Cancer Imaging
Archive.
"""

import os

import polars as pl
import torch
from torch.utils import data
from torch.utils.data import random_split
from torchvision import transforms
from pydicom import dcmread

from datasets import ROIDataset
from models import UNet
from utils import load_dicom_image
import losses
import trainers

img_size = (512, 512)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

FULL_IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((523, 316), interpolation=transforms.InterpolationMode.BICUBIC,antialias=True),
        transforms.Grayscale(num_output_channels=1),
    ]
)
ROI_IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((1298, 902), antialias=True),
        transforms.Grayscale(num_output_channels=1),
    ]
)


def _main():
    """Test the new functions."""
    # pipeline = CuratedBreastCancerROIPipeline(
    #   root="data/CBIS-DDSM/", img_labels={"roi": "ROI mask", "full": "full mammogram"}
    # )
    # df__pi, df__roi, df__mask = pipeline.start()
    # df__pi.write_csv('data/CBIS-DDSM/paired_image_set.csv')
    # df__roi.write_csv("data/CBIS-DDSM/roi_paired_image_set.csv")
    # df__mask.write_csv("data/CBIS-DDSM/mask_paired_image_set.csv")
    # TODO: Filter the ROI mask images so that the first image is chosen.
    df = pl.read_csv("data/CBIS-DDSM/roi_paired_image_set.csv")
    df = pl.read_csv("data/CBIS-DDSM/mask_paired_image_set.csv")
    # get_image_size_metrics(df)
    img_data = ROIDataset(
        df,
        "path",
        "path_right",
        img_transforms=FULL_IMAGE_TRANSFORM,
        roi_transform=FULL_IMAGE_TRANSFORM,
    )  # TODO: Create your own transforms for image size.
    train_size = int(0.7 * len(img_data))
    val_size = len(img_data) - train_size
    train_set, val_set = random_split(img_data, [train_size, val_size])
    train_loader = data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=8)
    val_loader = data.DataLoader(val_set, batch_size=2, shuffle=True, num_workers=4)
    #loss = torch.nn.MSELoss()
    loss = losses.TverskyLoss()
    model = UNet(in_channels=1, features=[16, 32, 64, 128])
    print(model)
    opt = torch.optim.Adam(params=model.parameters())
    trainer = trainers.MaskTrainer(model, opt, loss, gpu=True)
    trainer.train(train_loader, epochs=10)
    model = trainer.get_model()
    torch.save(model.state_dict(), 'models/unet_v1.pt')


def get_image_size_metrics(df: pl.DataFrame):
    """Get some basic metrics for observing images."""
    dims__full_image__str = df.select("image_dim").to_series().to_list()
    dims__roi_image__str = df.select("roi_dim").to_series().to_list()
    dims__full_image__tuple = [eval(value) for value in dims__full_image__str]
    heights__full_image = [h for h, w, d in dims__full_image__tuple]
    width__full_image = [w for h, w, d in dims__full_image__tuple]
    print(
        f"average height of full image: {sum(heights__full_image) / len(heights__full_image)}"
    )
    print(
        f"average width of full image: {sum(width__full_image ) / len(width__full_image)}"
    )
    print(f"max height of full image: {max(heights__full_image)}")
    print(f"max width of full image: {max(width__full_image)}")
    print(f"min height of full image: {min(heights__full_image)}")
    print(f"min width of full image: {min(width__full_image)}")
    print("")
    dims__roi_image__tuple = [eval(value) for value in dims__roi_image__str]
    heights__roi_image = [h for h, w, d in dims__roi_image__tuple]
    width__roi_image = [w for h, w, d in dims__roi_image__tuple]
    print(
        f"average height of roi image: {sum(heights__roi_image) / len(heights__roi_image)}"
    )
    print(
        f"average width of roi image: {sum(width__roi_image) / len(width__roi_image)}"
    )
    print(f"max height of roi image: {max(heights__roi_image)}")
    print(f"max width of roi image: {max(width__roi_image)}")
    print(f"min width of roi image: {min(width__roi_image)}")
    print(f"min width of roi image: {min(width__roi_image)}")


class DataPipeline:
    """Base Class for Data Related Pipelines.

    This will contain the base functions related to extracting data. The Data
    Pipeline will imitate PyTorch's Dataset class. The Pipeline will extract,
    transform, and load data into a PyTorch dataset and save a version of the
    dataset within the csv format. In the case that there are certain files
    (i.e. images) that will be used to train the machine learning models, then
    the paths to said files will be saved within the csv file itself together
    with any metadata found within any other possible csv file within the root.

    Parameters
    ----------
    root : String
        Path to the folder containing all of the data.

    """

    def __init__(self, root: str):
        """Init the Data Pipeline."""
        self.root = root
        self._extract_paths()
        self._label_paths()

    def start(self):
        """Activate and begin the Process for the Pipeline."""
        raise NotImplementedError(
            "start Method must be implemented before activating pipeline."
        )

    def _extract_paths(self):
        """Extract the path to files within root."""
        raw_paths = list("paths\n")
        for path, subdirs, files in os.walk(
            self.root
        ):  # TODO: Fix issue with csv files being included.
            paths = [os.path.join(path, name) for name in files if ".csv" not in name]
            raw_paths.extend(paths)
        self.files = raw_paths
        return self

    def _label_paths(self):
        """Label the file type of all files found within root."""
        all_files = list()
        for path in self.files:
            if "paths\n" in path:
                pass
            start = path.find(".")
            all_files.append({"path": path, "filetype": path[start + 1 :]})
        self.files_by_type = all_files
        return self


class CuratedBreastCancerClassifierPipeline(DataPipeline):
    """Pipeline for the CBIS-DDSM Dataset."""

    def __init__(self, root: str, labels: dict, cols: list[str]):
        """Init the Pipeline."""
        super().__init__(root)
        self.labels = labels
        self.cols = cols

    def start(self):
        """Start the pipeline processs."""
        print("starting Pipeline...")
        print("Writing dataset...")
        if os.path.isfile("data/CBIS-DDSM/dataset.csv"):
            df__dataset = pl.read_csv("data/CBIS-DDSM/dataset.csv")
        else:
            self.extract_paths()
            self.label_paths()
            df__descriptions = self.concat_description_sets()
            df__descriptions = self.create_unique_ids(
                df__descriptions, self.labelled_data
            )
            df__dataset = self.merge_data(self.labelled_data, df__descriptions, "UID")
            df__dataset.write_csv("data/CBIS-DDSM/dataset.csv")
        self.df_set = df__dataset
        print("Pipeline complete.")
        return df__dataset

    def label_paths(
        self,
    ):
        """Create labels for categorizing paths and extracting unique ID.

        Parameters
        ----------
        paths : List of Strings
            List of paths that direct to the relevant files.
        labels : Dictionary
            Contains the search term as its key and the label as its values.

        Returns
        -------
        List of Dictionaries
            Precursor to DataFrame that can be kept as is for saving as json,
            or any other configuration (allows flexibility to use polars or
            pandas).
        """
        data = list()
        for path in self.files:
            if "images" in path:
                for term, label in self.labels.items():
                    path = path
                    lpath = path.lower()
                    lterm = term.lower()
                    components = path.split("/")
                    raw_uid = components[3]
                    start = raw_uid.find("_") + 1
                    unique_id = raw_uid[start:]
                    if lterm in lpath:
                        data.append({"UID": unique_id, "path": path, "type": label})
                    else:
                        pass
            else:
                pass
        self.labelled_data = data
        return self

    def create_unique_ids(self, fname: str | pl.DataFrame) -> pl.DataFrame:
        """Create a column that contains a unique id created from other column values.

        Uses existing values from the dataset to develop a unique id that can then be
        used to merge two dataframes together. The intent is to match the id extracted
        from the path to the dicom file to the data found within the descriptive
        datasets.

        Parameters
        ----------
        fname : String or Polars DataFrame
            path to the csv file.
        cols : List of Strings
            column names

        Returns
        -------
        Polars DataFrame
            DataFrame containing the new unique id.
        """
        assert isinstance(fname, str) or isinstance(fname, pl.DataFrame), TypeError(
            "fname parameter must be either a string pointing to a csv file or a polar DataFrame."
        )
        if isinstance(fname, str):
            df = pl.read_csv(fname)
        else:
            df = fname
        df = df.with_columns(
            (
                pl.concat_str(
                    [pl.col(col) for col in self.cols],
                    separator="_",
                )
            ).alias("UID")
        )
        return df

    def extract_description_sets(self):
        """Extract all of the datasets describing the patients.

        Grabs all of the csv files containing the word 'description'
        within it and places them  within a list.
        """
        directory = os.listdir(self.root)
        descriptive_files = list()
        for fname in directory:
            if "description" in fname:
                descriptive_files.append(os.path.join(self.root, fname))
            else:
                pass
        self.desc_files = descriptive_files
        return self

    def concat_description_sets(self) -> pl.DataFrame:
        """Concatenate all of the datasets describing the patients.

        Grabs all of the csv files containing the word 'description'
        within it and concatenates them together into one descriptive
        dataset.
        """
        self.extract_description_sets()
        for i, fn in enumerate(self.desc_files):
            if i == 0:
                df = pl.read_csv(fn)
            else:
                df_concat = pl.read_csv(fn)
                df = pl.concat([df, df_concat], how="align")
        return df

    def merge_data(self, fname: str | pl.DataFrame, id: str) -> pl.DataFrame:
        """Merge the labeled dataset with paths to any other csv files."""
        assert isinstance(self.labelled_data, list) or isinstance(
            self.labelled_data, pl.DataFrame
        ), TypeError(
            "data parameter must be either a list of dictionaries or a polar DataFrame."
        )
        assert isinstance(fname, str) or isinstance(fname, pl.DataFrame), TypeError(
            "fname parameter must be either a string pointing to a csv file or a polar DataFrame."
        )
        if isinstance(self.labelled_data, list):
            df_paths = pl.DataFrame(self.labelled_data)
        else:
            df_paths = self.labelled_data
        if isinstance(fname, str):
            df_metadata = pl.read_csv(fname)
        else:
            df_metadata = fname
        df_merged = df_paths.join(df_metadata, on=id, how="inner")
        return df_merged


class CuratedBreastCancerROIPipeline(DataPipeline):
    """Pipeline For Creating ROI Dataset from the CBIS-DDSM Dataset.

    Pipeline that uses the CBIS-DDSM dataset to develop a machine learning
    model that zooms in on the region of interest of an image.

    Examples
    --------
    >pipeline = CuratedBreastCancerROIPipeline(
    >   root="data/CBIS-DDSM/", img_labels={"roi": "ROI mask", "full": "full mammogram"}
    >)
    >df__pi, df__roi, df__mask = pipeline.start()
    >df__pi.write_csv('data/CBIS-DDSM/paired_image_set.csv')
    >df__roi.write_csv("data/CBIS-DDSM/roi_paired_image_set.csv")
    >df__mask.write_csv("data/CBIS-DDSM/mask_paired_image_set.csv")
    """

    def __init__(self, root: str, img_labels: dict):
        """Init the class."""
        super().__init__(root)
        self.img_labels = img_labels
        self.link_images()
        self.create_unique_ids()

    def start(self):
        """Start the pipeline for data processing."""
        df__roi = pl.DataFrame(self.roidata)
        df__full_image = pl.DataFrame(self.fulldata)
        df__paired_images = df__full_image.join(df__roi, on="UID", how="left")
        # The below no longer works as the images are not necesarily in the correct order. Check to see if the dimensions are the same.
        df__roi_paired_images = df__paired_images.filter(
            pl.col("roi__series_description") == "cropped images"
        )
        df__mask_paired_images = df__paired_images.filter(
            pl.col("roi__series_description") == "ROI mask images"
        )
        return df__paired_images, df__roi_paired_images, df__mask_paired_images

    @staticmethod
    def roi_path_filter(path: str):
        """Get the roi paths from the file list."""
        if "roi" in path.lower():
            return True
        else:
            return False

    @staticmethod
    def full_image_path_filter(path: str):
        """Get the full image paths from the file list."""
        if "full" in path.lower():
            return True
        else:
            return False

    def create_unique_ids(self):
        """Create A unique id for linking images."""
        roidata = list()
        for rp in self.roi_files:
            ds_roi = dcmread(rp)
            roi_img = load_dicom_image(rp)
            roi_img_dim = roi_img.shape
            try:
                roi__series_description = ds_roi["SeriesDescription"].value
            except KeyError:
                roi__series_description = "Null"
            components = rp.split("/")
            raw_uid = components[3]
            start = raw_uid.find("_") + 1
            unique_id = raw_uid[start:-2]
            roidata.append(
                {
                    "UID": unique_id,
                    "path": rp,
                    "roi__series_description": roi__series_description,
                    "roi_width": roi_img_dim[0],
                    "roi_height": roi_img_dim[1],
                }
            )
        self.roidata = roidata
        fulldata = list()
        for fp in self.full_files:
            ds_full = dcmread(fp)
            full_img = load_dicom_image(fp)
            full_img_dim = full_img.shape
            try:
                full__series_description = ds_full["SeriesDescription"].value
            except KeyError:
                full__series_description = "Null"
            components = fp.split("/")
            raw_uid = components[3]
            start = raw_uid.find("_") + 1
            unique_id = raw_uid[start:]
            fulldata.append(
                {
                    "UID": unique_id,
                    "path": fp,
                    "full__series_description": full__series_description,
                    "full_width": full_img_dim[0],
                    "full_height": full_img_dim[1],
                }
            )
        self.fulldata = fulldata
        return self

    def link_images(self):
        """Connect ROI image to full image."""
        self.roi_files = list(filter(self.roi_path_filter, self.files))
        self.full_files = list(filter(self.full_image_path_filter, self.files))
        return self


if __name__ == "__main__":
    _main()
