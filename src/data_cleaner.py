import pandas as pd
class DataCleaner:
    """
    A class used to clean the crash data.

    Attributes
    ----------
    df : pd.DataFrame
        The dataframe containing crash data to be cleaned.
    min_latitude : float
        The minimum valid latitude.
    max_latitude : float
        The maximum valid latitude.
    max_speed_limit : int
        The maximum valid speed limit.
    """

    def __init__(self, df: pd.DataFrame, min_latitude: float = -90, max_latitude: float = 90,
                 max_speed_limit: int = 75):
        """
        Parameters
        ----------
        df : pd.DataFrame
            The dataframe containing crash data to be cleaned.
        min_latitude : float, optional
            The minimum valid latitude (default is -90).
        max_latitude : float, optional
            The maximum valid latitude (default is 90).
        max_speed_limit : int, optional
            The maximum valid speed limit (default is 75).
        """
        self.df = df
        self.min_latitude = min_latitude
        self.max_latitude = max_latitude
        self.max_speed_limit = max_speed_limit

    def clean_data(self) -> pd.DataFrame:
        """
        Cleans the crash data by performing the following steps:
        - Removing duplicates
        - Dropping rows with missing values in specified columns
        - Dropping unnecessary columns
        - Filtering latitude to be within valid range
        - Filtering out speed limits above a certain threshold
        - Resetting the index
        - Converting data types for specific columns

        Returns
        -------
        pd.DataFrame
            The cleaned dataframe.
        """
        # Remove duplicate rows
        self.df.drop_duplicates(inplace=True)

        # Drop rows with missing values in specified columns
        self.df.dropna(subset=[
            'NUM_UNITS', 'INJURIES_TOTAL', 'INJURIES_FATAL',
            'CRASH_HOUR', 'CRASH_DAY_OF_WEEK', 'CRASH_MONTH',
            'LATITUDE', 'LONGITUDE'
        ], inplace=True)

        # Drop unnecessary columns if they exist
        if 'LANE_CNT' in self.df.columns:
            self.df.drop(columns=['LANE_CNT'], inplace=True)
        if 'INTERSECTION_RELATED_I' in self.df.columns:
            self.df.drop(columns=['INTERSECTION_RELATED_I'], inplace=True)

        # Ensure latitude is within the valid range
        self.df = self.df[(self.df['LATITUDE'] >= self.min_latitude) & (self.df['LATITUDE'] <= self.max_latitude)]

        # Filter out speed limits above the specified maximum
        self.df = self.df[self.df['POSTED_SPEED_LIMIT'] < self.max_speed_limit]

        # Reset the index of the dataframe
        self.df.reset_index(inplace=True, drop=True)

        # Define the desired data types for specific columns
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

        # Convert columns to the specified data types
        self.df = self.df.astype(dtype_dict)

        return self.df
