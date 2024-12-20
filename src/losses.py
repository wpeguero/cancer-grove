"""Set of algorithms for calculating the loss."""
from typing import Optional, List

import torch
from torch import nn

def _main():
    """Test New Ideas."""
    pass

class TverskyLoss(nn.modules.loss._Loss):
    """Calculate the Tversky Loss."""

    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6):
        """Initialize the class."""
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        """Calculate based on input."""
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        tp = (y_true * y_pred).sum()
        fp = ((1 - y_true) * y_pred).sum()
        fn = (y_true * (1 - y_pred)).sum()

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky

### Deprecated losses from tensorflow
#class Dice(Loss):
#    """Dice Loss Algorithm.
#
#    Loss algorithm mainly used for calculating the
#    similarity between images.
#    """
#
#    def __init__(self, smooth=1e-6, gamma=2):
#        """Init the custom loss."""
#        super(Dice, self).__init__()
#        self.name = "NDL"
#        self.smooth = smooth
#        self.gamma = gamma
#
#    def call(self, y_true, y_pred):
#        """Logic for calculating the loss between y_true and y_pred."""
#        y_true, y_pred = (
#            tf.cast(y_true, dtype=tf.float32),
#            tf.cast(y_pred, dtype=tf.float32),
#        )
#        numerator = 2 * tf.reduce_sum(tf.multiply(y_pred, y_true)) + self.smooth
#        denominator = (
#            tf.reduce_sum(y_pred**self.gamma)
#            + tf.reduce_sum(y_true**self.gamma)
#            + self.smooth
#        )
#        result = 1 - tf.divide(numerator, denominator)
#        return result


#class Boundary(Loss):
#    """Boundary Variant Loss Algorithm.
#
#    Tasked with highlyy unbalanced segmentations. This
#    loss' form is that of a distance metric on space
#    contours and not regions. In this manner, it tackles
#    the problem posed by regional losses for highly
#    imbalanced segmentation tasks.
#    """
#
#    def __init__(self, theta0: int = 5, theta: int = 11):
#        """Init the custom loss."""
#        super(Boundary, self).__init__()
#        self.theta0 = theta0
#        self.theta = theta
#
#    def call(self, y_true, y_pred):
#        """Calculate the loss between the predicted values and the actual values."""
#        # Calculate the Boundary of the Image
#        y_tru = nn.softmax(y_true)
#        true_boundary = nn.max_pool2d(
#            1 - y_tru, ksize=self.theta0, strides=1, padding=(self.theta0 - 1) // 2
#        )
#        true_boundary -= 1 - y_tru
#
#        y_pre = nn.softmax(y_pred)
#        predicted_boundary = nn.max_pool2d(
#            1 - y_pre, ksize=self.theta0, strides=1, padding=(self.theta0 - 1) // 2
#        )
#        predicted_boundary -= 1 - y_pre
#
#        # Extended Boundary
#        true_extended_boundary = nn.max_pool2d(
#            1 - y_tru, ksize=self.theta, strides=1, padding=(self.theta - 1) // 2
#        )
#
#        predicted_extended_boundary = nn.max_pool2d(
#            1 - y_pre, ksize=self.theta, strides=1, padding=(self.theta - 1) // 2
#        )
#
#        # Calculate Precision and Recall
#        precision = tf.reduce_sum(
#            predicted_boundary * true_extended_boundary
#        ) / tf.reduce_sum(predicted_boundary)
#        recall = tf.reduce_sum(
#            predicted_extended_boundary * true_boundary
#        ) / tf.reduce_sum(true_boundary)
#
#        # Calculate the Boundary F1 Score and loss
#        BF1 = (2 * precision * recall) / (precision + recall)
#        loss = tf.reduce_mean(1 - BF1)
#        return loss


#class Tversky(Loss):
#    """Tversky Loss Algorithm."""
#
#    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-5, reduction="mean"):
#        """Init the custom loss."""
#        super(Tversky, self).__init__()
#        self.alpha = alpha
#        self.beta = beta
#        self.smooth = smooth
#        self.reduction = reduction
#
#    def call(self, y_true, y_pred):
#        """Logic for calculating the loss between the true values and the predictions."""
#        if y_true.ndim > 3:
#            y_true_f = y_true.reshape(y_true.shape[0], -1)
#            y_pred_f = y_pred.reshape(y_pred.shape[0], -1)
#        else:
#            y_true_f = y_true.flatten()
#            y_pred_f = y_pred.flatten()
#
#        intersection = tf.reduce_sum(y_true_f * y_pred_f)
#        tversky = (intersection + self.smooth) / (
#            intersection
#            + self.alpha * (tf.reduce_sum(y_pred_f * (1 - y_true_f)))
#            + self.beta * (tf.reduce_sum((1 - y_pred_f) * y_true_f))
#            + self.smooth
#        )
#        if self.reduction == "mean":
#            tversky = tversky.mean()
#        return tversky


if __name__ == "__main__":
    _main()
