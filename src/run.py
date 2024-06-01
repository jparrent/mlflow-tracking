import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from data_loader import DataLoader
from data_cleaner import DataCleaner
from feature_engineer import FeatureEngineer
from model_trainer import ModelTrainer


def main():
    """
    Main function to load, clean, engineer features, train, evaluate, and log the model.
    """
    # Set up directories
    current_directory: Path = Path.cwd()
    home_directory: Path = current_directory.parent
    data_directory: Path = home_directory / 'data/raw'
    processed_data_directory: Path = home_directory / 'data/processed'
    plots_directory: Path = home_directory / 'plots'

    # Create the plots directory if it doesn't exist
    os.makedirs(plots_directory, exist_ok=True)

    # Load or preprocess the data
    processed_data_file = processed_data_directory / 'processed_data.csv'
    if processed_data_file.exists():
        # Load processed data if it exists
        df = pd.read_csv(processed_data_file)
    else:
        # Load the raw data
        data_loader = DataLoader(data_directory)
        df: pd.DataFrame = data_loader.load_data('chicago_traffic_data_slim.csv')

        # Inspect the raw data
        print("Columns in raw data:")
        print(df.columns)

        # Clean the data
        data_cleaner = DataCleaner(df)
        df_clean: pd.DataFrame = data_cleaner.clean_data()

        # Feature engineering
        feature_engineer = FeatureEngineer(df_clean)
        df_features: pd.DataFrame = feature_engineer.engineer_features()

        # Save processed data
        df_features.to_csv(processed_data_file, index=False)

    # Encode categorical features
    df_encoded, target_map = feature_engineer.encode_categorical_features(df_features, 'DAMAGE')

    # Balance the dataset
    X_resampled, y_resampled = feature_engineer.balance_data(df_encoded, 'DAMAGE_CODE')

    # Prepare for modeling
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Initialize the model trainer
    model_trainer = ModelTrainer(X_train, X_test, y_train, y_test, plots_directory, target_map)

    # Train and evaluate the model
    best_params, model = model_trainer.train_model()
    model_trainer.evaluate_model(model, best_params)

    # Log the model with MLflow
    model_trainer.log_model(model, best_params, 'DAMAGE', 'initial_model')


if __name__ == '__main__':
    main()
