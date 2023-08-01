"""Set of algorithms for calculating the loss."""

import tensorflow as tf
from tensorflow import nn
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

class Boundary(Loss):
    """Boundary Variant Loss Algorithm

    Tasked with highlyy unbalanced segmentations. This
    loss' form is that of a distance metric on space
    contours and not regions. In this manner, it tackles
    the problem posed by regional losses for highly
    imbalanced segmentation tasks.
    """
    def __init__(self, theta0:int=5, theta:int=11):
        super(Boundary, self).__init__()
        self.theta0 = theta0
        self.theta = theta

    def call(self, y_true, y_pred):
        # Calculate the Boundary of the Image
        y_tru = nn.softmax(y_true)
        true_boundary = nn.max_pool2d(1 - y_tru, ksize=self.theta0, strides=1, padding=(self.theta0 - 1) // 2)
        true_boundary -= 1 - y_tru

        y_pre = nn.softmax(y_pred)
        predicted_boundary = nn.max_pool2d(1 - y_pre, ksize=self.theta0, strides=1, padding=(self.theta0 - 1) // 2)
        predicted_boundary -= 1 - y_pre

        # Extended Boundary
        true_extended_boundary = nn.max_pool2d(1 - y_tru, ksize=self.theta, strides=1, padding=(self.theta - 1) // 2)

        predicted_extended_boundary = nn.max_pool2d(1 - y_pre, ksize=self.theta, strides=1, padding=(self.theta - 1) // 2)

        # Calculate Precision and Recall
        precision = tf.reduce_sum(predicted_boundary * true_extended_boundary) / tf.reduce_sum(predicted_boundary)
        recall = tf.reduce_sum(predicted_extended_boundary * true_boundary) / tf.reduce_sum(true_boundary)

        # Calculate the Boundary F1 Score and loss
        BF1 = (2 * precision * recall) / (precision + recall)
        loss = tf.reduce_mean(1 - BF1)
        return loss


if __name__ == "__main__":
    _main()
