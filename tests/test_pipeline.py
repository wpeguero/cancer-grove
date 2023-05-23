"""Module for testing the pipeline library."""
from src.pipeline import *
from numpy import ndarray
import pytest
import pandas as pd

fn__clean_dataset = "data/CBIS-DDSM/fully_clean_dataset.csv"
file = "data/CBIS-DDSM/CBIS-DDSM/Calc-Test_P_00038_LEFT_CC/08-29-2017-DDSM-NA-96009/1.000000-full mammogram images-63992/1-1.dcm"

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
    sample_size = 500
    df = pd.read_csv(fn__clean_dataset)
    df_bal = balance_data(df, sample_size=sample_size)
    assert len(df_bal) == pytest.approx(sample_size, abs=0.01*sample_size)

def test_load_training_data():
    """Tests whether the load_training_data function loads the data for model training."""
    df = pd.read_csv(fn__clean_dataset)
    df = df.sample(1000, random_state=42)
    df__train = load_training_data(df, pathcol="Full Location")
    assert len(df__train == pytest.approx(1_000, abs=0.01*1_000))


if __name__ == "__main__":
    pytest.main()
