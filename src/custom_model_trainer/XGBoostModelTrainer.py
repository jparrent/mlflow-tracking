# src/custom_model_trainer/XGBoostModelTrainer.py

from model_trainer import ModelTrainer
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import pprint

class XGBoostModelTrainer(ModelTrainer):
    """
    A class used to train an XGBoost classifier and log the results using MLflow.

    Attributes
    ----------
    df_encoded_resampled : pd.DataFrame
        The DataFrame containing the resampled and encoded data.
    target : str
        The target column name.
    model_run_name : str
        The name of the model run for MLflow logging.
    plots_directory : str, optional
        The directory to save plots, by default None.
    params : dict
        The parameters for grid search.
    args : dict
        Additional arguments for the XGBoost classifier.
    """

    def __init__(self, df_encoded_resampled: pd.DataFrame, target: str, model_run_name: str,
                 plots_directory: str, params: dict, args: dict):
        self.df_encoded_resampled = df_encoded_resampled
        self.target = target
        self.model_run_name = model_run_name
        self.plots_directory = plots_directory
        self.params = params
        self.args = args

    def balance_data(self):
        """
        Applies SMOTE to balance the specified columns.
        """
        X = self.df_encoded_resampled.drop(columns=[self.target])
        y = self.df_encoded_resampled[self.target]

        smote = SMOTE(sampling_strategy='auto', k_neighbors=9, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        self.df_encoded_resampled = pd.concat([X_resampled, y_resampled], axis=1)

    def train_model(self):
        """
        Trains an XGBoost classifier and logs the results using MLflow.
        """
        X = self.df_encoded_resampled.drop(columns=[self.target + '_CODE', self.target])
        y = self.df_encoded_resampled[self.target + '_CODE']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        xgb = XGBClassifier(**self.args)

        grid_search = GridSearchCV(estimator=xgb, param_grid=self.params, cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)

        self.best_params = grid_search.best_params_
        print('Best Parameters:')
        pprint.pprint(self.best_params)

        eval_set = [(X_test, y_test), (X_train, y_train)]  # Define eval_set for early stopping

        self.final_model = XGBClassifier(**self.best_params)
        self.final_model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

        y_pred_test = self.final_model.predict(X_test)
        y_pred_train = self.final_model.predict(X_train)

        print("Test Data:")
        print(classification_report(y_test, y_pred_test))
        print("Train Data:")
        print(classification_report(y_train, y_pred_train))

        # Set these attributes here after training and evaluating the model
        self.X_test, self.y_test = X_test, y_test
        self.X_train, self.y_train = X_train, y_train
