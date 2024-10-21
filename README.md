# pyblight

Library for developing and processing cancer-based machine learning
models and data.

## Helpful Links

### Links to Informative journals about models

[Inception-v4, Inception-ResNet and the Impact of Residual connections on Learning](https://arxiv.org/pdf/1602.07261v2.pdf)

[Implementation of Inception-v4 and Inception-ResNet](https://github.com/zhulf0804/Inceptionv4_and_Inception-ResNetv2.PyTorch/blob/master/model/inceptionv4.py#L160)

### Links to Possible Datasets

[Duke-Breast-CanceR-MRI](https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri/)

[ACRIN-Contralateral-Breast-MR](https://www.cancerimagingarchive.net/collection/acrin-contralateral-breast-mr/)

[CBIS-DDSM](https://www.cancerimagingarchive.net/collection/cbis-ddsm/)

[CMMD | The Chinese Mammography Database](https://www.cancerimagingarchive.net/collection/cmmd/)

## TO DO LIST

### Pipeline for Finding Region Of Interest

- [ ] Create a dataset that pairs the raw image to the mask
- [ ] create a dataset that pairs the raw image and the cropped image
- [ ] Develop a model that is able to train on developing the mask
- [ ] Create a model that is able to train on extracting a cropped image.
