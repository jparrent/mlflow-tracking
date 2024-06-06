import os
from pathlib import Path
import pandas as pd
from custom_data_loader import create_data_loader
from custom_data_cleaner import create_data_cleaner
from custom_feature_engineer import create_feature_engineer
from custom_model_trainer import create_model_trainer
import yaml
import time

# Read and parse the YAML configuration file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

def main():
    target = 'DAMAGE'
    model_run_name = 'initial_model'

    # Set up directories
    current_directory: Path = Path.cwd()
    home_directory: Path = current_directory.parent
    data_directory: Path = home_directory / 'data/raw'
    processed_data_directory: Path = home_directory / 'data/processed'
    plots_directory: Path = home_directory / 'plots'

    # Create the plots directory if it doesn't exist
    os.makedirs(plots_directory, exist_ok=True)

    # Load or preprocess the data
    processed_data_file = processed_data_directory / 'processed_data.pkl'
    if processed_data_file.exists():
        # Load processed data if it exists
        print('Loading already cleaned data.')
        df_features = pd.read_pickle(processed_data_file)
    else:
        # Dynamically import and use data loader based on configuration
        for data_loader in config.get('data_loaders', []):
            loader_instance = create_data_loader(data_loader)
            print("Loading Data...")
            start_time = time.time()
            df = loader_instance(data_directory / 'chicago_traffic_data_slim.csv').load_data()
            end_time = time.time()
            print(f"Data loaded using {data_loader}. Time taken: {end_time - start_time:.2f} seconds")

        # Inspect the raw data
        print(f"Columns in raw data: {df.columns}")

        # Dynamically import and use data cleaner based on configuration
        for data_cleaner in config.get('data_cleaners', []):
            cleaner_instance = create_data_cleaner(data_cleaner)
            print("Cleaning Data...")
            start_time = time.time()
            df = cleaner_instance(df).clean_data()
            end_time = time.time()
            print(f"Data cleaned using {data_cleaner}. Time taken: {end_time - start_time:.2f} seconds")

        # Dynamically import and use feature engineer based on configuration
        for feature_engineer in config.get('feature_engineers', []):
            feature_engineer_instance = create_feature_engineer(feature_engineer)
            print('Engineering Features...')
            start_time = time.time()
            df_features = feature_engineer_instance(df, target).engineer_features()
            end_time = time.time()
            print(f"Feature engineering completed. Time taken: {end_time - start_time:.2f} seconds")

        # Save processed data
        print('Saveing Processed Data...')
        df_features.to_pickle(processed_data_file)

    # Dynamically import and use model trainer based on configuration
    for model_trainer in config.get('model_trainers', []):
        model_trainer_instance = create_model_trainer(model_trainer)
        print('Training Model...')
        start_time = time.time()
        model_trainer_instance(df_features, target, model_run_name, plots_directory).train_model()
        end_time = time.time()
        print(f"Modeling completed. Time taken: {end_time - start_time}")

if __name__ == '__main__':
    # Start MLFlow before running
    # mlflow server --host ###.#.#.# --port #### --backend-store-uri sqlite:///model_tracking.db
    main()