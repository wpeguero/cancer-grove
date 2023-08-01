# Cancer Grove

This is a project for developing a machine learning algorithm using The Cancer Imaging Archive (TCIA) data sets that will
1. Classify the image as either benign or malignant
2. Create an ROI around the possible location of the tumor
3. determine the stage of the tumor.

GroveAI is the culmination of functions used to extract the data and develop the machine learning models. The goal is to develop algorithms that can be used for extracting, analyzing and formatting both images and metadata to feed into Artificial Neural Networks.

## Losses

The losses library is a new experimental set of algorithms made for calculating the loss in training the machine learning models. These will include algorithms that are able to compare the actual image to the predicted image for masks.

The latest resource for listing the types of losses for image segmentation is [neptune](https://neptune.ai/blog/image-segmentation)
