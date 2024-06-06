from abc import ABC, abstractmethod
from dataclasses import dataclass
import pandas as pd

@dataclass
class BaseDataCleaner(ABC):
    """
    An abstract base class for data cleaning operations.
    """

    df: pd.DataFrame

    @abstractmethod
    def clean_data(self):
        """
        Abstract method to clean the data.
        """
        pass

    @staticmethod
    def reset_indices(df):
        """Resets the indices of the DataFrame."""
        df.reset_index(inplace=True, drop=True)