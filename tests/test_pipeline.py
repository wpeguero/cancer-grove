"""Module for testing the pipeline library."""
from src.pipeline import *
from numpy import ndarray
import pytest
import polars as pl
from torch.utils import data


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
    seed = 0
    n = 500
    df = pl.DataFrame({
        "groups": (pl.int_range(0, n, eager=True) % 5).shuffle(seed=seed),
        "values": pl.int_range(0, n, eager=True).shuffle(seed=seed)
        })
    df_bal = balance_data(df, sample_size=sample_size, columns=['groups'])
    assert len(df_bal) == pytest.approx(sample_size, abs=0.01*sample_size)


if __name__ == "__main__":
    pytest.main()
