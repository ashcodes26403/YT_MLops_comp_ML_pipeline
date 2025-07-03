import os
import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier
import yaml

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_building.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters loaded successfully')
        return params
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    :param file_path: Path to the CSV file
    :return: Loaded DataFrame
    """
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s with shape %s', file_path, df.shape)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def train_model(X_train, y_train, model_params: dict) -> RandomForestClassifier:
    """Train a Random Forest classifier."""

    try:
        if(X_train.shape[0] != y_train.shape[0]):
            raise ValueError("X_train and y_train must have the same number of rows")

        clf = RandomForestClassifier(n_estimators=model_params['n_estimators'], random_state=model_params['random_state'])
        clf.fit(X_train, y_train)
        logger.debug('Model trained successfully')

        return clf

    except ValueError as e:
        logger.error('ValueError: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def save_model(model, model_path: str) -> None:
    """Save the trained model to a file."""
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)  # Create the directory if it doesn't exist
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('Model saved to %s', model_path)
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while saving the model: %s', e)
        raise

def main():
    try:
        params = {'n_estimators': 50, 'random_state': 42}
        train_data = load_data('./data/processed/train_tfidf.csv')
        X_train = train_data.drop('label', axis=1)
        y_train = train_data['label']
        #model_params = load_params(params)

        train_data_path = './data/processed/train_tfidf.csv'
        train_data = pd.read_csv(train_data_path)
        X_train = train_data.drop('label', axis=1)
        y_train = train_data['label']

        model = train_model(X_train, y_train, params)

        model_path = 'models/model.pkl'
        save_model(model, model_path)
        logger.debug('Model saved to %s', model_path)

    except FileNotFoundError as e:     
        logger.error('File not found: %s', e)
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

if __name__ == '__main__':
    main()