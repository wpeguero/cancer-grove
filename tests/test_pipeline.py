"""Module for testing the pipeline library."""
from src.pipeline import *
from numpy import ndarray
import pytest
import pandas as pd

file = "tests/samples/DBCMRI_BMRI2_1-030.dcm"
sample_tag_names = [
    'PatientID',
    'PatientSex'
]

def test_extract_data_output():
    """Test the extraction process of the data."""
    datapoint = extract_data(file)
    assert type(datapoint) == dict

#def test_transform_data(): # This will require rework.
#    """Test whether the extracted data has been successfully transformed."""
#    datapoint = extract_data(file)
#    datapoint = transform_data(datapoint)
#    for key, value in datapoint.items():
#        if (key == 'Subject ID') or (key == 'image'):
#            pass
#        else:
#            assert type(value) == int

def test_rescale_image():
    """Test whether images are rescaled appropriately."""
    datapoint = extract_data(file)
    img = rescale_image(datapoint.get('image'))
    assert type(img) == ndarray
    ishape = img.shape
    oshape = datapoint['image'].shape
    factor = oshape[0] / ishape[0]
    assert factor == 2

def test_data_balance():
    """Tests whether the data_balance function evenly balances the 12 data  groups."""
    filename = "data/DBCMRI/segmentation_filepath_mapping.csv"
    sample_size = 500
    df = pd.read_csv(filename)
    df_bal = balance_data(df, columns=['Segmentation Label'],sample_size=sample_size)
    assert len(df_bal) == pytest.approx(sample_size, abs=0.01*sample_size)

if __name__ == "__main__":
    pytest.main()
