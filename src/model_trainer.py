from abc import ABC, abstractmethod
from dataclasses import dataclass
import pandas as pd

@dataclass
class ModelTrainer(ABC):
    """
    A base class used to train a model from a DataFrame.

    Attributes
    ----------
    df : pd.DataFrame
        The DataFrame containing the data to be passed to trainer.
    """

    df: pd.DataFrame

    @abstractmethod
    def train_model(self):
        """
        Abstract method to train the model
        """
        pass