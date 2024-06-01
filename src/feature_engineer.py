import pandas as pd
import numpy as np
from geopy.distance import geodesic
from imblearn.over_sampling import SMOTE


class FeatureEngineer:
    """
    A class used to engineer features from a DataFrame.

    Attributes
    ----------
    df : pd.DataFrame
        The DataFrame containing the data to be processed.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the data to be processed.
        """
        self.df = df

    def engineer_features(self) -> pd.DataFrame:
        """
        Engineer new features for the DataFrame.

        Returns
        -------
        pd.DataFrame
            The DataFrame with the engineered features.
        """
        # Define the coordinates of the Chicago city center
        city_center = (41.881832, -87.623177)

        # Calculate the distance from each crash to the Chicago city center
        self.df['distance_from_chicago'] = self.df.apply(
            lambda row: geodesic((row['LATITUDE'], row['LONGITUDE']), city_center).miles, axis=1
        )

        # Filter out crashes that are more than 100 miles from the city center
        self.df = self.df[self.df['distance_from_chicago'] < 100]

        # Encode cyclic features for time-related columns
        self.encode_cyclic_feature('CRASH_HOUR')
        self.encode_cyclic_feature('CRASH_DAY_OF_WEEK')
        self.encode_cyclic_feature('CRASH_MONTH')

        # Apply KMeans clustering to the geographic data
        self.kmeans_clustering()

        # Drop the original longitude and latitude columns
        self.df.drop(columns=['LONGITUDE', 'LATITUDE'], axis=1, inplace=True)

        return self.df

    def encode_cyclic_feature(self, column_name: str):
        """
        Encode a time-related feature as cyclic using sine and cosine transformations.

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
        Apply KMeans clustering to the geographic data.

        Parameters
        ----------
        n_clusters : int, optional
            The number of clusters to form, by default 9.
        """
        from sklearn.cluster import KMeans

        # Extract the longitude and latitude values
        X = self.df[['LONGITUDE', 'LATITUDE']].values

        # Fit the KMeans model
        kmeans = KMeans(n_clusters=n_clusters, random_state=123)
        self.df['latlon_cluster_num'] = kmeans.fit_predict(X)

        # Convert cluster labels to categorical type
        self.df['latlon_cluster_num'] = self.df['latlon_cluster_num'].astype('category')

    def encode_categorical_features(self, df: pd.DataFrame, target: str):
        """
        Encode categorical features using one-hot encoding.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the data to be encoded.
        target : str
            The target column name to be encoded.

        Returns
        -------
        pd.DataFrame
            The DataFrame with the encoded categorical features.
        dict
            A mapping of the target column's categories to codes.
        """
        # Map target categories to codes
        target_map = {
            '$500 OR LESS': 0,
            '$501 - $1,500': 1,
            'OVER $1,500': 2
        }

        # Encode the target column
        df[target + '_CODE'] = df[target].map(target_map)

        df_encoded = pd.DataFrame()

        # Encode categorical columns using one-hot encoding
        for col, dtype in df.dtypes.items():
            if dtype == 'category' and col != target:
                encoded_col = pd.get_dummies(df[col], prefix=col)
                df_encoded = pd.concat([df_encoded, encoded_col], axis=1)
            else:
                df_encoded[col] = df[col]

        return df_encoded, target_map

    def balance_data(self, df: pd.DataFrame, target: str):
        """
        Balance the dataset using SMOTE.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the data to be balanced.
        target : str
            The target column name.

        Returns
        -------
        pd.DataFrame
            The resampled feature data.
        pd.Series
            The resampled target data.
        """
        # Separate features and target
        X = df.drop(columns=[target])
        y = df[target]

        # Apply SMOTE to balance the data
        smote = SMOTE(sampling_strategy='auto', k_neighbors=9, random_state=123)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        return X_resampled, y_resampled
