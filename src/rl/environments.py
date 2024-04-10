"""Environments Module.

Contains all of the custom environments for building machine learning
models.
"""

import torchrl
import tensordict

def _main():
    """Test classes."""
    pass


class HangmanEnvironment(torchrl.envs.EnvBase):
    """Pytorch Implementation of the Hangman Environment.

    This is an implementation based on the TensorFlow Hangman
    Environment made for the job application.

    Arguments
    ---------
    words : list
        Contains a list of words for playing hangman.
    """

    def __init___(self, words:list):
        """Init the class."""
        super().__init__(device=device, batch_size=[])

    def _step(self, td:tensordict.TensorDictBase):
        """Compute the Next Step for the actor.

        Parameters
        ----------
        td: TensorDictBase
            Contains the data required for calculating the next step.
        """
        pass


if __name__ == "__main__":
    _main()
