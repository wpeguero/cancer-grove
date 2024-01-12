"""Set of Classes for Training Machine Learning Models."""
import torch
from torch import nn, optim
from torch.utils import data

class TrainModel:
    """Class for training pytorch machine learning models.

    This class functions as an environment for training the
    pytorch models.

    Parameters
    ----------
    model : torch Module
        Model which will be trained within this class.
    optimizer : torch Optimizer
        optimizer used to change the weights on the machine
        learning model.
    loss : torch Loss
        The chosen loss to compare the prediction and the target.
    """

    def __init__(self, model:nn.Module, optimizer:optim.Optimizer, loss):
        """Initialize the class."""
        self.model = model
        self.opt = optimizer
        self.criterion = loss

    def get_model(self):
        """Get the Model post training.

        Returns
        -------
        torch Module
            The model at any point in time before or after training.
        """
        return self.model

    def train(self, trainloader:data.DataLoader, epochs:int, gpu=False):
        """Train the machine learning model.

        Uses the given model, optimizer, and loss provided when the
        class was initizalized in conjunction with the trainloader to
        train the given model. The model is trained based on the
        number of epochs provided. For each epoch, the model is
        trained by iterating through the dataset and the model
        weights are updated based on the loss.

        Parameters
        ----------
        trainloader : torch DataLoader
            dataset loaded and ready for model training.

        epochs : int
            number of times the model loops through the batched set.

        gpu : bool
            Determines whether the gpu is used to train dataset.
        """
        if  gpu == True:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")
        print("The model will be running on ", device, "device")
        self.model.to(device)
        steps_per_epoch = len(trainloader.dataset) // trainloader.batch_size
        print("Starting Training.")
        for epoch in range(epochs):
            running_loss = 0.0
            self.model.train(True)
            for i, (inputs, labels) in enumerate(trainloader, 0):
                inputs = inputs.to(device)
                labels = labels.to(device)
                self.opt.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.opt.step()

                running_loss += loss.item()
                if i+1 == steps_per_epoch:
                    print(f'[{epoch + 1:3d}/{epochs}, {i + 1:5d}] loss: {running_loss / steps_per_epoch:.3f}')
