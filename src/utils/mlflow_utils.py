# src/utils/mlflow_utils.py

import mlflow
import mlflow.xgboost

def log_mlflow(model, best_params, X_test, y_test, X_train, y_train, model_run_name, plots_directory, metric):
    mlflow.set_tracking_uri('http://127.0.0.1:8080')
    mlflow.set_experiment(f"/{model_run_name}")

    with mlflow.start_run():
        mlflow.log_params(best_params)
        mlflow.log_metrics({
            "accuracy_test": model.score(X_test, y_test),
            "accuracy_train": model.score(X_train, y_train),
        })
        mlflow.xgboost.log_model(model, "xgboost_model.json")
        if plots_directory:
            mlflow.log_artifact(plots_directory / f"{model_run_name}_{metric}_curve.png",
                                artifact_path=f"{model_run_name}_{metric}_curve.png")
