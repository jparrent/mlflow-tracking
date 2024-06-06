from dataclasses import dataclass
from abc import ABC, abstractmethod
import pandas as pd

@dataclass
class DataLoader(ABC):
    """
    An abstract class for loading data.
    """
    file_path: str

    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """
        Abstract method to load data into a pandas DataFrame.
        """
        pass