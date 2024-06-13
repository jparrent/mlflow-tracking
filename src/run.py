import os
from pathlib import Path
import pandas as pd
from custom_data_loader import create_data_loader
from custom_data_cleaner import create_data_cleaner
from custom_feature_engineer import create_feature_engineer
from custom_model_trainer import create_model_trainer
from custom_plotting import create_plotting
from utils.mlflow_utils import log_mlflow
import yaml
import time

def main():
    # Read and parse the YAML configuration file
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Set up directories
    current_directory: Path = Path.cwd()
    home_directory: Path = current_directory.parent
    data_directory: Path = home_directory / 'data/raw'
    processed_data_directory: Path = home_directory / 'data/processed'
    plots_directory: Path = home_directory / 'plots'

    # Create the plots directory if it doesn't exist
    os.makedirs(plots_directory, exist_ok=True)

    # Create the processed data directory if it doesn't exist
    os.makedirs(processed_data_directory, exist_ok=True)

    # Define the target variable
    target = 'DAMAGE'

    # Load or preprocess the data
    processed_data_file = processed_data_directory / 'processed_data.pkl'
    if processed_data_file.exists():
        print('Loading already cleaned data.')
        df_features = pd.read_pickle(processed_data_file)
    else:
        # Load the data using the specified data loader
        start_time = time.time()
        data_loader = create_data_loader(config['data_loaders'][0]['name'])
        data_loader_params = config['data_loaders'][0]['params']
        data_loader_instance = data_loader(data_directory / data_loader_params['file_name'])
        print("Loading Data...")
        df = data_loader_instance.load_data()
        print(f"Data loading took {time.time() - start_time:.2f} seconds.")

        # Clean the data using the specified data cleaner
        start_time = time.time()
        data_cleaner_instance = create_data_cleaner(config['data_cleaners'][0]['name'])
        print("Cleaning Data...")
        df_cleaned = data_cleaner_instance(df).clean_data()
        print(f"Data cleaning took {time.time() - start_time:.2f} seconds.")

        # Engineer features using the specified feature engineer
        start_time = time.time()
        feature_engineer_instance = create_feature_engineer(config['feature_engineers'][0]['name'])
        print("Engineering Features...")
        df_features = feature_engineer_instance(df_cleaned, target).engineer_features()
        print(f"Feature engineering took {time.time() - start_time:.2f} seconds.")

        # Save processed data
        print("Saving Processed Data...")
        df_features.to_pickle(processed_data_file)
    # Train the model using the specified model trainer
    model_trainer_config = config['model_trainers'][0]
    model_run_name = config['model_run_name']
    model_trainer = create_model_trainer(model_trainer_config['name'])
    model_trainer_instance = model_trainer(df_features, target, model_run_name, plots_directory, model_trainer_config['params'], model_trainer_config['args'])

    start_time = time.time()
    print("Balancing Data...")
    model_trainer_instance.balance_data()
    print(f"Data balancing took {time.time() - start_time:.2f} seconds.")

    start_time = time.time()
    model_trainer_instance.train_model()
    print(f"Model training took {time.time() - start_time:.2f} seconds.")

    # Plot the results if specified
    metric = config['plotting'][0]['params']['metric']
    if 'plotting' in config:
        for plot in config['plotting']:
            plotter = create_plotting(plot['name'])
            plotter_instance = plotter()
            plotter_instance.plot_curve(model_trainer_instance.final_model.evals_result(), model_run_name, plots_directory, plot['params']['metric'])

    # Log the results to MLflow
    start_time = time.time()
    log_mlflow(model_trainer_instance.final_model, model_trainer_instance.best_params, model_trainer_instance.X_test, model_trainer_instance.y_test, model_trainer_instance.X_train, model_trainer_instance.y_train,
               model_run_name, plots_directory, metric)
    print(f"Logging to MLflow took {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
