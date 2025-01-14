"""Set of algorithms for calculating the loss."""
from typing import Optional, List

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


if __name__ == "__main__":
    _main()
