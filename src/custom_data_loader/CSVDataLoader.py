from data_loader import DataLoader
import pandas as pd

class CSVDataLoader(DataLoader):

    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.file_path)
        # Additional custom loading logic if needed
        return df