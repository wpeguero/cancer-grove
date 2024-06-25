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
# import utils

img_size = (512, 512)
torch.manual_seed(42)
torch.cuda.manual_seed(42)


def _main():
    """Test the new functions."""
    # dir = 'data/CBIS-DDSM/images/'
    # fpaths = utils.get_file_paths(dir, 'data/CBIS-DDSM/paths.csv')
    lb = {'mask': 'mask', 'cropped':'crop', 'full':'full'}
    pipe = Pipeline(root='data/CBIS-DDSM/', labels=lb)
    pipe.start()
    exit()
    df__paths = pl.read_csv("data/CBIS-DDSM/paths.csv")
    pathset = list()
    for row in df__paths.iter_rows(named=True):
        fpath = row["paths"]
        fpath = fpath[:-1]
        components = fpath.split("/")
        raw_uid = components[3]
        start = raw_uid.find('_') + 1
        unique_id = raw_uid[start:]
        if "ROI mask" in fpath:
            label = "mask"
        elif "cropped images" in fpath:
            label = "crop"
        elif "full mammogram images" in fpath:
            label = "full"
        else:
            label = ""
        pathset.append({"paths": fpath, "UID": unique_id, "img type": label})
    df__nupaths = pl.DataFrame(pathset)
    df__nupaths.write_csv("data/CBIS-DDSM/nupaths.csv")


class Pipeline:
    """Pipeline for the CBIS-DDSM Dataset."""

    def __init__(self, root: str, labels: dict):
        """Init the Pipeline."""
        self.root = root
        self.labels = labels

    def start(self):
        """Start the pipeline processs."""
        print("starting Pipeline...")
        paths = self.extract_paths(self.root)
        labeled_paths = self.label_paths(paths, self.labels)
        df__descriptions = self.concat_description_sets()
        df__descriptions = self.create_unique_id(df__descriptions, labeled_paths)
        df__dataset = self.merge_data(labeled_paths, df__descriptions, 'UID')
        print("Writing dataset...")
        if os.path.isfile("data/CBIS-DDSM/dataset.csv"):
            df__dataset = pl.read_csv("data/CBIS-DDSM/dataset.csv")
        else:
            df__dataset.write_csv('data/CBIS-DDSM/dataset.csv')
        print("Pipeline complete.")
        return df__dataset

    @staticmethod
    def extract_paths(root: str) -> list[str]:
        """Extract the paths to all files within root directory."""
        all_files = list()
        all_files.append("paths\n")
        for path, subdirs, files in os.walk(root):
            for name in files:
                all_files.append(os.path.join(path, name, "\n"))
        return all_files

    @staticmethod
    def label_paths(paths: list[str], labels: dict) -> list[dict]:
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
        for path in paths:
            if 'images' in path:
                for term, label in labels.items():
                    path = path[:-1]
                    lpath = path.lower()
                    lterm = term.lower()
                    components = path.split("/")
                    raw_uid = components[3]
                    start = raw_uid.find('_') + 1
                    unique_id = raw_uid[start:]
                    if lterm in lpath:
                        data.append({"UID": unique_id, "path": path, "type": label})
                    else:
                        pass
            else:
                pass
        return data

    @staticmethod
    def create_unique_id(fname: str | pl.DataFrame, cols: list[str]) -> pl.DataFrame: # TODO: Deprecate the cols parameter
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
                    pl.col("patient_id"),
                    pl.col("left or right breast"),
                    pl.col("image view"),
                    pl.col("abnormality id"),
                    separator="_"
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

    @staticmethod
    def merge_data(
        data: list[dict] | pl.DataFrame, fname: str | pl.DataFrame, id: str
    ) -> pl.DataFrame:
        """Merge the labeled dataset with paths to any other csv files."""
        assert isinstance(data, list) or isinstance(
            data, pl.DataFrame
        ), TypeError(
            "data parameter must be either a list of dictionaries or a polar DataFrame."
        )
        assert isinstance(fname, str) or isinstance(fname, pl.DataFrame), TypeError(
            "fname parameter must be either a string pointing to a csv file or a polar DataFrame."
        )
        if isinstance(data, list):
            df_paths = pl.DataFrame(data)
        else:
            df_paths = data
        if isinstance(fname, str):
            df_metadata = pl.read_csv(fname)
        else:
            df_metadata = fname
        df_merged = df_paths.join(df_metadata, on=id, how="inner")
        return df_merged


if __name__ == "__main__":
    _main()
