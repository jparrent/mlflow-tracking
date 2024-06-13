from feature_engineer import FeatureEngineer
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from sklearn.cluster import KMeans


class AccidentFeatureEngineer(FeatureEngineer):
    def __init__(self, df: pd.DataFrame, target: str):
        self.df = df.copy()  # Make a copy to avoid modifying the original DataFrame
        self.target = target

    def engineer_features(self):
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
        city_center = (41.881832, -87.623177)
        self.df['distance_from_chicago'] = self.df.apply(
            lambda row: geodesic((row['LATITUDE'], row['LONGITUDE']), city_center).miles, axis=1
        )

    def remove_outliers(self):
        self.df = self.df[self.df['distance_from_chicago'] < 100].copy()  # Ensure to copy to avoid warnings

    def encode_cyclic_features(self, column_name: str):
        radians = 2. * np.pi / self.df[column_name].max()
        self.df.loc[:, f'{column_name}_sin'] = np.sin(self.df[column_name] * radians)
        self.df.loc[:, f'{column_name}_cos'] = np.cos(self.df[column_name] * radians)

    def kmeans_clustering(self, n_clusters: int = 9):
        X = self.df[['LONGITUDE', 'LATITUDE']].values
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.df['latlon_cluster_num'] = kmeans.fit_predict(X)
        self.df.loc[:, 'latlon_cluster_num'] = self.df['latlon_cluster_num'].astype('category')

    def drop_lon_lat_columns(self):
        self.df.drop(columns=['LONGITUDE', 'LATITUDE'], inplace=True)

    def encode_categorical_features(self):
        target_map = {
            '$500 OR LESS': 0,
            '$501 - $1,500': 1,
            'OVER $1,500': 2
        }
        self.df.loc[:, self.target + '_CODE'] = self.df[self.target].map(target_map).astype('category')

        df_to_encode = self.df.drop(
            columns=[self.target, self.target + '_CODE', 'RD_NO', 'CRASH_DATE', 'CRASH_HOUR', 'CRASH_DAY_OF_WEEK',
                     'CRASH_MONTH'])
        df_encoded = pd.get_dummies(df_to_encode, columns=df_to_encode.select_dtypes(include=['category']).columns)

        self.df = pd.concat([self.df[[self.target, self.target + '_CODE']], df_encoded], axis=1)
