from abc import ABC, abstractmethod
from dataclasses import dataclass
import pandas as pd

@dataclass
class FeatureEngineer(ABC):
    """
    A base class used to engineer features from a DataFrame.

    Attributes
    ----------
    df : pd.DataFrame
        The DataFrame containing the data to be processed.
    """

    df: pd.DataFrame

    @abstractmethod
    def engineer_features(self):
        """
        Abstract method to engineer new features for the DataFrame.
        """
        pass