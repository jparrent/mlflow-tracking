import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from pathlib import Path


class ModelTrainer:
    """
    A class used to train an XGBoost classifier and log the model with MLflow.

    Attributes
    ----------
    data : pd.DataFrame
        The DataFrame containing the feature and target data.
    target : str
        The name of the target column.
    drop_columns : list
        A list of column names to drop from the data.
    plots_directory : Path, optional
        The directory where plot images will be saved, by default None.
    """

    def __init__(self, data: pd.DataFrame, target: str, drop_columns: list, plots_directory: str = None):
        """
        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame containing the feature and target data.
        target : str
            The name of the target column.
        drop_columns : list
            A list of column names to drop from the data.
        plots_directory : str, optional
            The directory where plot images will be saved, by default None.
        """
        self.data = data
        self.target = target
        self.drop_columns = drop_columns
        self.plots_directory = Path(plots_directory) if plots_directory else Path.cwd() / 'plots'
        self.target_map = {
            '$500 OR LESS': 0,
            '$501 - $1,500': 1,
            'OVER $1,500': 2
        }
        self.data[target + '_CODE'] = self.data[target].map(self.target_map)

        for col in self.drop_columns:
            if col in self.data.columns:
                self.data.drop(columns=[col], inplace=True)
            else:
                print(f'Column {col} absent')

        self.X = self.data.drop(columns=[self.target + '_CODE'])  # Features
        self.y = self.data[self.target + '_CODE']  # Target

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        self.plots_directory.mkdir(parents=True, exist_ok=True)

    def train_model(self, reg_alpha: float = 0, early_stopping: bool = True):
        """
        Train an XGBoost classifier using grid search and optionally early stopping.

        Parameters
        ----------
        reg_alpha : float, optional
            L1 regularization term on weights, by default 0.
        early_stopping : bool, optional
            Whether to use early stopping, by default True.
        """
        xgb = XGBClassifier(enable_categorical=True, reg_alpha=reg_alpha, eval_metric='mlogloss')

        params = {
            'max_depth': [15, 20, 25, 30],
            'learning_rate': [0.1, 0.01],
            'n_estimators': [300, 350, 400, 450]
        }

        grid_search = GridSearchCV(estimator=xgb, param_grid=params, cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(self.X_train, self.y_train)

        best_params = grid_search.best_params_
        print('Best Parameters:')
        print(best_params)

        eval_set = [(self.X_test, self.y_test), (self.X_train, self.y_train)]  # Include train data in eval_set

        if early_stopping:
            early_stopping_rounds = 10
            final_model = XGBClassifier(**best_params, early_stopping_rounds=early_stopping_rounds)
            final_model.fit(self.X_train, self.y_train, eval_set=eval_set, verbose=False)
        else:
            final_model = XGBClassifier(**best_params)
            final_model.fit(self.X_train, self.y_train, eval_set=eval_set, verbose=False)

        self.evaluate_model(final_model, best_params)
        self.log_model(final_model, best_params, self.target, 'xgboost_model')

    def evaluate_model(self, model, best_params):
        """
        Evaluate the trained model.

        Parameters
        ----------
        model : XGBClassifier
            The trained XGBoost classifier.
        best_params : dict
            The best hyperparameters found during training.
        """
        y_pred = model.predict(self.X_test)
        report = classification_report(self.y_test, y_pred, target_names=self.target_map.keys())
        print(report)

        # Plot and save feature importance
        self.plot_feature_importance(model, best_params)

    def plot_feature_importance(self, model, best_params):
        """
        Plot and save the feature importances.

        Parameters
        ----------
        model : XGBClassifier
            The trained XGBoost classifier.
        best_params : dict
            The best hyperparameters found during training.
        """
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(10, 6))
        plt.title('Feature Importances')
        plt.bar(range(self.X_train.shape[1]), importances[indices], align='center')
        plt.xticks(range(self.X_train.shape[1]), self.X_train.columns[indices], rotation=90)
        plt.tight_layout()
        plot_path = self.plots_directory / 'feature_importance.png'
        plt.savefig(plot_path)
        plt.close()
        print(f'Feature importance plot saved to {plot_path}')

    def log_model(self, model, best_params, target, model_name):
        """
        Log the trained model with MLflow.

        Parameters
        ----------
        model : XGBClassifier
            The trained XGBoost classifier.
        best_params : dict
            The best hyperparameters found during training.
        target : str
            The name of the target column.
        model_name : str
            The name to assign to the logged model.
        """
        with mlflow.start_run() as run:
            mlflow.xgboost.log_model(model, model_name)
            mlflow.log_params(best_params)
            mlflow.log_metric('accuracy', model.score(self.X_test, self.y_test))
            mlflow.log_artifacts(self.plots_directory)
            mlflow.log_param('target', target)
        print(f'Model logged with run_id: {run.info.run_id}')
