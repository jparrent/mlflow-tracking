from model_trainer import ModelTrainer
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_curve, auc
import mlflow
import mlflow.xgboost
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
    """

    def __init__(self, df_encoded_resampled: pd.DataFrame, target: str, model_run_name: str,
                 plots_directory: str = None):

        self.df_encoded_resampled = df_encoded_resampled
        self.target = target
        self.model_run_name = model_run_name
        self.plots_directory = plots_directory

    def balance_data(self):
        """
        Applies SMOTE to balance the specified columns.
        """
        # Separate features and target variable
        X = self.df_encoded_resampled.drop(columns=[self.target])
        y = self.df_encoded_resampled[self.target]

        # Apply SMOTE to balance the data
        smote = SMOTE(sampling_strategy='auto', k_neighbors=9, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        self.df_encoded_resampled = pd.concat([X_resampled, y_resampled], axis=1)

    def train_model(self, reg_alpha: float = 0, early_stopping: bool = False):
        """
        Trains an XGBoost classifier and logs the results using MLflow.

        Parameters
        ----------
        reg_alpha : float, optional
            The L1 regularization term, by default 0.
        early_stopping : bool, optional
            Whether to use early stopping, by default False.
        """

        X = self.df_encoded_resampled.drop(columns=[self.target + '_CODE', self.target])  # Features
        y = self.df_encoded_resampled[self.target + '_CODE']  # Target

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        xgb = XGBClassifier(enable_categorical=True, reg_alpha=reg_alpha, eval_metric='mlogloss')

        params = {
            'max_depth': [10],
            'learning_rate': [0.1, 0.01],
            'n_estimators': [450, 500]
        }

        grid_search = GridSearchCV(estimator=xgb, param_grid=params, cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        print('Best Parameters:')
        pprint.pprint(best_params)

        eval_set = [(X_test, y_test), (X_train, y_train)]  # Include train data in eval_set

        if early_stopping:
            early_stopping_rounds = 10
            final_model = XGBClassifier(**best_params, early_stopping_rounds=early_stopping_rounds)
            final_model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
            early_stopping_rounds = final_model.best_iteration
        else:
            final_model = XGBClassifier(**best_params)
            final_model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

        y_pred_test = final_model.predict(X_test)
        y_pred_train = final_model.predict(X_train)

        print("Test Data:")
        print(classification_report(y_test, y_pred_test))
        print("Train Data:")
        print(classification_report(y_train, y_pred_train))

        # Plot AUC Learning Curves for both test and train data if plots_directory is provided
        if self.plots_directory:
            results = final_model.evals_result()
            epochs = len(results['validation_0']['mlogloss'])
            x_axis = range(0, epochs)

            fig, ax = plt.subplots()
            ax.plot(x_axis, results['validation_0']['mlogloss'], label='Test', color='orange')
            ax.plot(x_axis, results['validation_1']['mlogloss'], label='Train', color='blue')
            ax.legend()
            plt.ylabel('mlogloss')
            plt.title('XGBoost mlogloss')
            plt.savefig(self.plots_directory / f"{self.model_run_name}_mlogloss_curve.png")

        # MLflow Tracking
        mlflow.set_tracking_uri('http://127.0.0.1:8080')
        mlflow.set_experiment(f"/{self.model_run_name}")

        with mlflow.start_run():
            mlflow.log_params(best_params)
            mlflow.log_metric("xgb_reg_alpha", reg_alpha)
            mlflow.log_metric("accuracy_test", final_model.score(X_test, y_test))
            mlflow.log_metric("accuracy_train", final_model.score(X_train, y_train))
            mlflow.log_param("early_stopping_used", early_stopping)
            if early_stopping:
                mlflow.log_param("early_stopping_rounds", early_stopping_rounds)
            mlflow.xgboost.log_model(final_model, "xgboost_model.json")
            if self.plots_directory:
                mlflow.log_artifact(self.plots_directory / f"{self.model_run_name}_mlogloss_curve.png",
                                    artifact_path=f"{self.model_run_name}_mlogloss_curve.png")
