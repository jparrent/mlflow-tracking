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

This will load the data, clean it, engineer features, train a model, and save the processed data and model.

## Scripts
### DataLoader

`
src/data_loader.py
`

Loads the raw data from the specified directory.

### DataCleaner

`
src/data_cleaner.py
`

Cleans the data by handling missing values, removing duplicates, and filtering invalid data. The cleaning steps are configurable and serve as a placeholder for more advanced data cleaning techniques.

### FeatureEngineer

`
src/feature_engineer.py
`

Engineers new features from the raw data, such as calculating distances from a fixed point and encoding cyclic features. It also includes clustering based on latitude and longitude. This script is designed to be customizable for different feature engineering needs.

### ModelTrainer

`
src/model_trainer.py
`

Trains a machine learning model using XGBoost. It includes hyperparameter tuning, model evaluation, and logging with MLflow. The trainer is designed to be adaptable for different models and evaluation metrics.
run.py

The main script that ties together the data loading, cleaning, feature engineering, and model training steps. It saves the processed data to disk and loads it if it already exists to avoid redundant processing.
Data

The data used in this project is a placeholder obtained from a recent interview to set up a larger personal ML/DS workflow. The goal is to create a flexible framework that can be easily adapted to different datasets and machine learning tasks. The current data cleaning and feature engineering steps are basic and meant to be expanded upon for more complex projects.

## Future Work

- Custom Algorithms: Make it easier to swap out algorithms and customize the ModelTrainer.
- Advanced Data Cleaning: Expand the DataCleaner to include more sophisticated cleaning techniques.
- Enhanced Feature Engineering: Improve the FeatureEngineer with more advanced feature extraction methods.
- Modularization: Further modularize the code to enhance reusability and flexibility.

## Contributions

Feel free to contribute to this project by opening issues or submitting pull requests. Your feedback and suggestions are welcome!

## License

This project is licensed under the MIT License. See the LICENSE file for details.