import pandas as pd
from pathlib import Path


class DataLoader:
    """
    A class used to load data from a specified directory.

    Attributes
    ----------
    data_directory : Path
        The directory where the data files are stored.
    """

    def __init__(self, data_directory: Path):
        """
        Parameters
        ----------
        data_directory : Path
            The directory where the data files are stored.
        """
        self.data_directory = data_directory

    def load_data(self, file_name: str) -> pd.DataFrame:
        """
        Loads data from a CSV file into a pandas DataFrame.

        Parameters
        ----------
        file_name : str
            The name of the CSV file to load.

        Returns
        -------
        pd.DataFrame
            The data loaded from the CSV file.
        """
        # Construct the full file path
        csv_file_path = self.data_directory / file_name

        # Load the CSV file into a pandas DataFrame and return it
        return pd.read_csv(csv_file_path)
