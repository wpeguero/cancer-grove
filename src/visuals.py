"""Set of functions and algorithms to observe model training."""

from pydicom import dcmread
import plotly.express as px
import numpy as np


def _main():
    fn__image = "data/CBIS-DDSM/images/Calc-Test_P_00038_LEFT_CC/08-29-2017-DDSM-NA-96009/1.000000-full mammogram images-63992/1-1.dcm"
    fn__image_full = "data/CBIS-DDSM/images/Calc-Test_P_00038_LEFT_CC_1/08-29-2017-DDSM-NA-94942/1.000000-ROI mask images-18515/1-1.dcm"
    fn__image_mask = "data/CBIS-DDSM/images/Calc-Test_P_00038_LEFT_CC_1/08-29-2017-DDSM-NA-94942/1.000000-ROI mask images-18515/1-2.dcm"
    ds = dcmread(fn__image)
    fig = px.imshow(ds.pixel_array/np.max(ds.pixel_array))
    fig.show()
    ds_full = dcmread(fn__image_full)
    fig_full = px.imshow(ds_full.pixel_array/np.max(ds.pixel_array))
    fig_full.show()
    print(ds.pixel_array.shape)
    ds_mask = dcmread(fn__image_mask)
    fig_mask = px.imshow(ds_mask.pixel_array/np.max(ds.pixel_array))
    fig_mask.show()


if __name__ == "__main__":
    _main()
