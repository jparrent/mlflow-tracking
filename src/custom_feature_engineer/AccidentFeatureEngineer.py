from feature_engineer import FeatureEngineer
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from sklearn.cluster import KMeans

class AccidentFeatureEngineer(FeatureEngineer):
    """
    A class used to engineer features from a DataFrame.

    Attributes
    ----------
    df : pd.DataFrame
        The DataFrame containing the data to be processed.
    """

    def __init__(self, df: pd.DataFrame, target: str):
        """
        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the data to be processed.
        """
        self.df = df
        self.target = target

    def engineer_features(self):
        """
        Engineer new features for the DataFrame.

        Parameters
        ----------
        target : str
            The target column name.
        """
        self.calculate_distance_from_chicago()
        self.remove_outliers()
        self.kmeans_clustering()
        self.drop_lon_lat_columns()

        self.encode_cyclic_features('CRASH_HOUR')
        self.encode_cyclic_features('CRASH_DAY_OF_WEEK')
        self.encode_cyclic_features('CRASH_MONTH')
        self.encode_categorical_features()

        return self.df


    def calculate_distance_from_chicago(self):
        """
        Calculates the distance of each data point from the city center (Chicago).
        """
        # City center coordinates for Chicago
        city_center = (41.881832, -87.623177)

        # Calculate the distance from each crash to the Chicago city center
        self.df['distance_from_chicago'] = self.df.apply(
            lambda row: geodesic((row['LATITUDE'], row['LONGITUDE']), city_center).miles, axis=1
        )

    def remove_outliers(self):
        """
        Removes outliers based on a distance threshold of 100 miles from the city center.
        """
        self.df = self.df[self.df['distance_from_chicago'] < 100]

    def encode_cyclic_features(self, column_name: str):
        """
        Encodes a cyclic feature using sine and cosine transformations.

        Parameters
        ----------
        column_name : str
            The name of the column to be encoded.
        """
        radians = 2. * np.pi / self.df[column_name].max()
        self.df[f'{column_name}_sin'] = np.sin(self.df[column_name] * radians)
        self.df[f'{column_name}_cos'] = np.cos(self.df[column_name] * radians)

    def kmeans_clustering(self, n_clusters: int = 9):
        """
        Applies KMeans clustering to the geographic data.

        Parameters
        ----------
        n_clusters : int, optional
            The number of clusters to form, by default 9.
        """
        # Extract the longitude and latitude values
        X = self.df[['LONGITUDE', 'LATITUDE']].values

        # Perform KMeans clustering with specified clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.df['latlon_cluster_num'] = kmeans.fit_predict(X)

        # Convert cluster numbers to categorical data
        self.df['latlon_cluster_num'] = self.df['latlon_cluster_num'].astype('category')

    def drop_lon_lat_columns(self):
        """Drops the original LONGITUDE and LATITUDE columns."""
        self.df.drop(columns=['LONGITUDE', 'LATITUDE'], axis=1, inplace=True)

    def encode_categorical_features(self):
        """
        Encodes categorical features using one-hot encoding.

        Parameters
        ----------
        target : str
            The target column name.
        """
        target_map = {
            '$500 OR LESS': 0,
            '$501 - $1,500': 1,
            'OVER $1,500': 2
        }
        self.df[self.target + '_CODE'] = self.df[self.target].map(target_map)

        # Create a copy of the DataFrame excluding the target column and the newly created target code column
        df_to_encode = self.df.drop(columns=[self.target, self.target + '_CODE', 'RD_NO', 'CRASH_DATE', 'CRASH_HOUR', 'CRASH_DAY_OF_WEEK', 'CRASH_MONTH'])

        # Apply one-hot encoding to the remaining categorical columns
        df_encoded = pd.get_dummies(df_to_encode, columns=df_to_encode.select_dtypes(include=['category']).columns)

        # Concatenate the one-hot encoded columns with the original DataFrame
        self.df = pd.concat([self.df[[self.target, self.target + '_CODE']], df_encoded], axis=1)