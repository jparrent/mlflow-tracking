# MLflow Tracking Project

This project is a Machine Learning and Data Science pipeline aimed at predicting outcomes based on various features using MLflow for tracking experiments. The current setup serves as a flexible and reusable framework that can be adapted and expanded for different datasets and machine learning tasks. The main purpose is to establish a flexible and reusable workflow for personal ML/DS projects.

## Project Structure

The project directory structure is as follows:

```plaintext
mlflow-tracking/
├── data/
│   ├── processed/
│   └── raw/
│       └── placeholder_data.csv
├── notebooks/
├── src/
│   ├── data_cleaner.py
│   ├── custom_data_cleaner/
│   │   ├── __init__.py
│   │   └── AccidentDataCleaner.py
│   ├── data_loader.py
│   ├── custom_data_loader/
│   │   ├── __init__.py
│   │   └── CSVDataLoader.py
│   ├── feature_engineer.py
│   ├── custom_feature_engineer/
│   │   ├── __init__.py
│   │   └── AccidentFeatureEngineer.py
│   ├── model_trainer.py
│   ├── custom_model_trainer/
│   │   ├── __init__.py
│   │   └── XGBoostModelTrainer.py
│   ├── custom_plotting/
│   │   ├── __init__.py
│   │   └── Plotting.py
│   ├── run.py
│   └── config.yaml
├── plots/
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Installation

To get started, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/jparrent/mlflow-tracking.git
cd mlflow-tracking
pip install -r requirements.txt
```

## Usage

To run the project, execute the main script run.py:

```bash
python src/run.py
```

This will load the data, clean it, engineer features, train a model, and save the processed data and model, with tracking in MLFlow.

## Scripts

### DataLoader (Base Class)

`src/data_loader.py`

This script contains the DataLoader class, which is responsible for loading raw data from the specified directory. This serves as a base class for any custom data loading logic you might want to implement.

### CSVDataLoader

`src/custom_data_loader/CSVDataLoader.py`

This script contains the CSVDataLoader class, which inherits from DataLoader and implements the logic to load data specifically from CSV files. It demonstrates how to extend the base data loading functionality for specific file types.

### DataCleaner (Base Class)

`src/data_cleaner.py`

This script defines a base DataCleaner class. The AccidentDataCleaner class inherits from this base class and can be extended for more specific cleaning operations. This approach allows for a flexible and modular data cleaning pipeline.

### AccidentDataCleaner

`src/custom_data_cleaner/AccidentDataCleaner.py`

This script contains the AccidentDataCleaner class, which extends the data cleaning functionalities specific to accident data. It handles missing values, removes duplicates, and filters invalid data.

### FeatureEngineer (Base Class)

`src/feature_engineer.py`

This script defines the FeatureEngineer class, which is responsible for feature engineering tasks such as calculating distances from a fixed point, encoding cyclic features, and clustering based on latitude and longitude. This class serves as a base class that can be customized for different feature engineering needs.

### AccidentFeatureEngineer

`src/custom_feature_engineer/AccidentFeatureEngineer.py`

This script contains the AccidentFeatureEngineer class, which inherits from FeatureEngineer and implements additional feature engineering steps specific to accident data. It demonstrates how to extend the base feature engineering functionality for specific datasets.

### ModelTrainer (Base Class)

`src/model_trainer.py`

This script contains the ModelTrainer class, which trains machine learning models. This class includes methods for handling the training process, hyperparameter tuning, model evaluation, and logging with MLflow. It serves as a flexible base for implementing specific training algorithms.

### XGBoostModelTrainer

`src/custom_model_trainer/XGBoostModelTrainer.py`

### Plotting (Custom Module)

`src/custom_plotting/Plotting.py`

This script contains custom plotting utilities that can be used to visualize various aspects of the data, model performance, or any other relevant metrics. It enhances the project's capability to generate insightful visualizations as part of the data exploration or model evaluation process.

### Run/Main

`src/run.py`

This script contains the XGBoostModelTrainer class, which inherits from ModelTrainer and implements the logic to train a machine learning model using XGBoost. It includes hyperparameter tuning, model evaluation, and logging with MLflow. This class is designed to be adaptable for different models and evaluation metrics.
run.py


The main script that ties together the data loading, cleaning, feature engineering, and model training steps. It orchestrates the entire workflow and ensures that each step is executed in the correct order. It saves the processed data to disk and loads it if it already exists to avoid redundant processing.

## Data

The data used in this project is a placeholder obtained from a recent interview to set up a larger personal ML/DS workflow. The goal is to create a flexible framework that can be easily adapted to different datasets and machine learning tasks. The current data cleaning and feature engineering steps are basic and meant to be expanded upon for more complex projects.

## Future Work

- Custom Algorithms: Make it easier to swap out algorithms and customize the ModelTrainer.
- Modularization: Further modularize the codebase to support easier integration of new components.

## Contributions

Feel free to contribute to this project by opening issues or submitting pull requests. Your feedback and suggestions are welcome!

## License

This project is licensed under the MIT License. See the LICENSE file for details.