from data_cleaner import BaseDataCleaner
from dataclasses import dataclass
import pandas as pd

@dataclass
class AccidentDataCleaner(BaseDataCleaner):
    """
    A custom data cleaner class for specific data cleaning operations.
    """

    def clean_data(self):
        """Cleans the crash data."""
        self.drop_columns()
        self.drop_duplicates()
        self.drop_missing_values()
        self.clean_intersection_related()
        self.clean_lon_lat()
        self.clean_speed_limit()
        self.reset_indices(self.df)
        self.update_dtypes()

        return self.df

    def drop_columns(self):
        """
        Drops specified columns from the DataFrame.
        """
        self.df.drop(columns=['LANE_CNT'], inplace=True)

    def drop_duplicates(self):
        """Drops duplicate rows from the DataFrame."""
        self.df.drop_duplicates(inplace=True)

    def drop_missing_values(self):
        """
        Drops rows with missing values in specified columns from the DataFrame.
        """
        self.df.dropna(subset=[
            'NUM_UNITS', 'INJURIES_TOTAL', 'INJURIES_FATAL',
            'CRASH_HOUR', 'CRASH_DAY_OF_WEEK', 'CRASH_MONTH',
            'LATITUDE', 'LONGITUDE'
        ], inplace=True)

    def clean_intersection_related(self):
        """
        Cleans the INTERSECTION_RELATED_I values in the DataFrame.
        """
        self.df['INTERSECTION_RELATED_I'] = self.df['INTERSECTION_RELATED_I'].replace('NO', 'N')
        self.df.drop(columns=['INTERSECTION_RELATED_I'], inplace=True)

    def clean_lon_lat(self):
        """
        Cleans the latitude and longitude values in the DataFrame.
        """
        latitude_max = 90
        latitude_min = -90
        self.df = self.df[(self.df['LATITUDE'] >= latitude_min) & (self.df['LATITUDE'] <= latitude_max)]

    def clean_speed_limit(self):
        """
        Cleans the 'POSTED_SPEED_LIMIT' column in the DataFrame.
        """
        max_speed_limit = 75
        self.df = self.df[self.df['POSTED_SPEED_LIMIT'] < max_speed_limit]

    def update_dtypes(self):
        """Updates the data types of specific columns in the DataFrame."""
        dtype_dict = {
            'CRASH_DATE': 'datetime64[ns]',
            'POSTED_SPEED_LIMIT': 'int64',
            'WEATHER_CONDITION': 'category',
            'LIGHTING_CONDITION': 'category',
            'FIRST_CRASH_TYPE': 'category',
            'ROADWAY_SURFACE_COND': 'category',
            'DAMAGE': 'category',
            'PRIM_CONTRIBUTORY_CAUSE': 'category',
            'SEC_CONTRIBUTORY_CAUSE': 'category',
            'NUM_UNITS': 'int64',
            'INJURIES_TOTAL': 'int64',
            'INJURIES_FATAL': 'int64',
            'CRASH_HOUR': 'int64',
            'CRASH_DAY_OF_WEEK': 'int64',
            'CRASH_MONTH': 'int64',
            'LATITUDE': 'float64',
            'LONGITUDE': 'float64'
        }

        self.df = self.df.astype(dtype_dict)