"""Custom Dataset Classes that Inherit from Pytorch's Dataset Class."""
import os

from torch.utils import data
import polars as pl


def _main():
    """Test the new functions."""
    pass


class ImageSet(data.Dataset):
    """Dataset extracted from paths to cancer images.

    Dataset subclass that will grab the path to a folder
    containing the entire set of images and if images are
    organized based on folders, then it will attach a label.
    The label will be numerical and represent the origin of the
    folder.

    *Alternatively, one can use the torchvision.data.ImageFolder class for the same reason.*
    """

    def __init__(self, root='train/', image_loader=None, transform=None):
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
        images = [self.loader(os.path.join(self.root, folder)) for folder in self.folders]
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

    def __init__(self, csvfile:str|pl.DataFrame, label_column:str='pathology', image_loader=None, image_transforms=None, cat_transforms=None):
        """Initialize the class."""
        if type(csvfile) == str:
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
        image = Image.open(self.csv['path'][index])
        if self.image_transforms:
            image = self.image_transforms(image)
        supplementary_data = np.array(self.csv.select(pl.exclude('path', self.lcol)))
        labels = np.array(self.csv.select(self.lcol))
        if self.cat_transforms:
            labels = self.label_transforms(labels)
        sample = {'image':image, 'supplementary data':supplementary_data, 'labels':labels}
        return sample


class DICOMSet(data.Dataset):
    """Dataset used to load and extract information from DICOM images.

    This custom dataset extracts the image from the DICOM file in
    conjunction with the selected values found within the DICOM file.
    The data is then brought together to create a singular datapoint
    to be used in training a machine learning model with mixed input.
    """

    pass


if __name__ == "__main__":
    _main()
