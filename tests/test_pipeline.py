"""Module for testing the pipeline library."""
from src.pipeline import *
from numpy import ndarray
import pytest
import pandas as pd
from torch.utils import data

fn__clean_dataset = "data/CBIS-DDSM/fully_clean_dataset.csv"
file = "data/CBIS-DDSM/CBIS-DDSM/Calc-Test_P_00038_LEFT_CC/08-29-2017-DDSM-NA-96009/1.000000-full mammogram images-63992/1-1.dcm"
fn__image_set = "data/Dataset_BUSI_with_GT/"

def test_image_dataset_class():
    img_set = ImageSet(root=fn__image_set)
    img_loader = data.DataLoader(img_set, batch_size=3)


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
    df_bal = balance_data(df, sample_size=sample_size, columns=['left or right breast','image view'])
    assert len(df_bal) == pytest.approx(sample_size, abs=0.01*sample_size)

def test_load_training_data():
    """Tests whether the load_training_data function loads the data for model training."""
    df = pd.read_csv(fn__clean_dataset)
    df = df.sample(1000, random_state=42)
    df__train = load_training_data(df, pathcol="Full Location")
    assert len(df__train == pytest.approx(1_000, abs=0.01*1_000))


if __name__ == "__main__":
    pytest.main()
