# Standard library imports
import logging

from pathlib import Path 

# Third-party imports
import pandas as pd
import yaml
from sklearn.utils._testing import ignore_warnings

# Local application/library specific imports
from src.data_preparation import DataPreparation
from src.model_training import ModelTraining

logging.basicConfig(level=logging.INFO)


@ignore_warnings(category=Warning)
def main():

    # Configuration file path
    config_path = "./src/config.yaml"

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Load CSV file into a DataFrame
    main_dir = Path(__file__).parent
    data_path = main_dir / "data/resale_transactions.csv"
    df = pd.read_csv(data_path)


    # Initialize and run data preparation
    data_prep = DataPreparation(config)
    cleaned_df = data_prep.clean_data(df)

    # Initialize model training with the created preprocessor
    model_training = ModelTraining(config, data_prep.preprocessor)

    # Split the data
    X_train, X_val, X_test, y_train, y_val, y_test = model_training.split_data(
        cleaned_df
    )

    # Train and evaluate baseline models with default hyperparameters
    baseline_models, baseline_metrics = (
        model_training.train_and_evaluate_baseline_models(
            X_train, y_train, X_val, y_val
        )
    )

    # Train and evaluate tuned models with hyperparameter tuning
    tuned_models, tuned_metrics = model_training.train_and_evaluate_tuned_models(
        X_train, y_train, X_val, y_val
    )

    # Combine all models and their metrics into dictionaries
    all_models = {**baseline_models, **tuned_models}
    all_metrics = {**baseline_metrics, **tuned_metrics}

    # Find the best model based on R² score
    best_model_name = max(all_metrics, key=lambda k: all_metrics[k]["R²"])
    best_model = all_models[best_model_name]
    logging.info(f"Best Model Found: {best_model_name}")

    # Evaluate the best model on the test set
    final_metrics = model_training.evaluate_final_model(
        best_model, X_test, y_test, best_model_name
    )


if __name__ == "__main__":
    main()
