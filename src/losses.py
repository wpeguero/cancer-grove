"""Set of algorithms for calculating the loss."""

import tensorflow as tf
from keras.losses import Loss

def _main():
    """Test New Ideas."""
    pass


class Dice(Loss):
    """Dice Loss Algorithm

    Loss algorithm mainly used for calculating the
    similarity between images.
    """
    def __init__(self, smooth=1e-6, gamma=2):
        super(Dice, self).__init__()
        self.name = 'NDL'
        self.smooth = smooth
        self.gamma = gamma

    def call(self, y_true, y_pred):
        """Logic for calculating the loss between y_true and y_pred."""
        y_true, y_pred = tf.cast(y_true, dtype=tf.float32), tf.cast(y_pred, dtype=tf.float32)
        numerator = 2 * tf.reduce_sum(tf.multiply(y_pred, y_true)) + self.smooth
        denominator = tf.reduce_sum(y_pred ** self.gamma) + tf.reduce_sum(y_true ** self.gamma) + self.smooth
        result = 1 - tf.divide(numerator, denominator)
        return result

class IntersectionOverUnion(Loss):
    """Intersection Over Union (IoU) Balanced Loss Algorithm

    Aims to increase the gradient of samples with high IoU
    and decrease the gradient of samples with low IoU. In
    this way the localization accuracy of machine learning
    models is increased.
    """
    pass

class Boundary(Loss):
    """Boundary Variant Loss Algorithm

    Tasked with highlyy unbalanced segmentations. This
    loss' form is that of a distance metric on space
    contours and not regions. In this manner, it tackles
    the problem posed by regional losses for highly
    imbalanced segmentation tasks.
    """
    pass

class Lovasz(Loss):
    """Lovasz-Softmax Loss

    Performs direct optimization of the mean intersection-
    over-union loss in neural  networks based on the convex
    lovasz extension of sub-modular losses.
    """
    pass


if __name__ == "__main__":
    _main()
