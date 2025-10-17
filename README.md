# HDB_Prices_RegressionEndToEnd

A machine learning project to predict HDB resale prices using data preprocessing and model training pipelines. Built with Python, this project leverages libraries like Pandas, Scikit-learn, and YAML for configuration.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Built with love by the HDB Prediction Team](#Built with love by the HDB Prediction Team)

## Features

- Data cleaning and preprocessing of HDB resale transaction data
- Feature engineering (e.g., converting story range to average, extracting lease information)
- Training and evaluation of baseline models (Linear Regression, Ridge, Lasso)
- Hyperparameter tuning for Ridge and Lasso models using GridSearchCV
- Model evaluation with metrics (MAE, MSE, RMSE, R² score)

## Installation

Follow these steps to set up the project locally:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/jonyling/hdb-resale-prediction.git
   cd hdb-resale-prediction

2. **Install dependencies**:

    ```bash
    pip install -r requirements.txt

3. **Set up configuration**:

    Ensure the config.yaml file is in the root directory with the correct path to your dataset (e.g., data/resale_transactions.csv).
    Update the target_column, val_size, and other parameters as needed.

4. **Run the project**:

    ```bash
    python main.py

## Usage

- Data Preparation: The data_preparation.py script cleans and preprocesses the HDB resale data, handling missing values, duplicates, and feature transformations.
- Model Training: The model_training.py script trains baseline models and performs hyperparameter tuning, evaluating models with various metrics.
- Main Execution: Run main.py to execute the full pipeline, including data splitting, model training, and evaluation.

## Configuration

The project uses a config.yaml file for configuration. Key parameters include:
    - file_path: Path to the dataset (e.g., data/resale_transactions.csv).
    - target_column: The column to predict (e.g., resale_price).
    - val_size: Validation set size (e.g., 0.2).
    - param_grid: Hyperparameter grid for Ridge and Lasso (e.g., alpha: [0.1, 1, 10, 100, 1000]).
    - scoring: Evaluation metric (e.g., r2).
    - numerical_features, nominal_features, ordinal_features: Lists of feature types for preprocessing.
Edit config.yaml to customize the pipeline for your dataset.

## Contributing

We welcome contributions! To contribute:
    Fork the repository.
    Create a new branch (git checkout -b feature/your-feature-name).
    Commit your changes (git commit -m 'Add your feature').
    Push to the branch (git push origin feature/your-feature-name).
    Open a Pull Request.
Please follow our Code of Conduct (CODE_OF_CONDUCT.md) and ensure code adheres to PEP 8 style guidelines.

## License

This project is licensed under the MIT License (LICENSE).

## Contact

For questions or feedback, reach out to:
    Email: <jonyling@hotmail.com>
    GitHub: jonyling
    X: @JonyLing1

## Built with love by the HDB Prediction Team

### Explanation of Sections

- **Project Name and Description**: Reflects the HDB resale price prediction focus and mentions key technologies.
- **Features**: Summarizes the data preparation, model training, and evaluation capabilities.
- **Installation**: Provides clear steps to clone, install dependencies, configure, and run the project.
- **Usage**: Describes the role of each script and expected output.
- **Configuration**: Highlights the `config.yaml` file and key configurable parameters.
- **Contributing**: Offers a standard contribution guide.
- **License and Contact**: Includes licensing and contact details for collaboration.
This README is concise yet informative, tailored to your project's structure. Adjust the repository URL, contact details, and dataset path as needed. Let me know if you'd like to expand any section!
