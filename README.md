# House Price Prediction — MLOps Assignment 1

**Student ID:** 23L-2640

## Project Overview

This project is a machine learning project for predicting house prices using the Kaggle House Prices dataset.

A Random Forest Regression model is trained using selected house features and evaluated using Mean Absolute Error (MAE) and R² score.

This project demonstrates basic MLOps practices using Git, GitHub, and Visual Studio Code for version control and project organization.

## Project Structure

```text
mlops-project-23L-2640/
│
├── data/
│   └── house_prices.csv
│
├── src/
│   └── train_23L-2640.py
│
├── model/
│   └── house_price_model.pkl
│
├── .gitignore
├── requirements.txt
└── README.md
```

The `data/` and `model/` directories are excluded from Git tracking using `.gitignore`. The raw dataset and trained model are therefore not uploaded to the public GitHub repository.

## Dataset

The project uses the **House Prices: Advanced Regression Techniques** dataset from Kaggle.

The dataset contains information about residential properties and their sale prices.

The following features are used for prediction:

* OverallQual
* GrLivArea
* GarageCars
* TotalBsmtSF
* FullBath
* YearBuilt

**Target variable:** `SalePrice`

## Machine Learning Model

The project uses a **Random Forest Regressor** from Scikit-learn.

The model is trained using an 80/20 train-test split.

The final model uses:

```text
n_estimators = 300
random_state = 42
```

StandardScaler normalization is also applied to the training and testing features.

## Installation

Clone the repository:

```bash
git clone https://github.com/abdullah-qasim-dev/mlops-project-23L-2640.git
```

Navigate to the project directory:

```bash
cd mlops-project-23L-2640
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

On Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

Run the training script from the project root directory:

```bash
python src/train_23L-2640.py
```

The script loads the dataset, preprocesses the features, trains the Random Forest model, evaluates its performance, and saves the trained model.

The trained model is saved to:

```text
model/house_price_model.pkl
```

## Model Evaluation

The model is evaluated using:

* Mean Absolute Error (MAE)
* R² Score

The baseline model achieved approximately:

```text
MAE: 19210.50
R² Score: 0.8908
```

The exact results may vary depending on the final model configuration.

## Version Control

Git was used for:

* Tracking project changes
* Creating commits
* Creating feature branches
* Practicing soft and hard resets
* Creating and resolving merge conflicts
* Using Git stash
* Maintaining the project history

The project contains the following feature branches:

```text
feature-preprocessing-23L-2640
feature-tuning-23L-2640
```

## GitHub Repository

Public repository:

https://github.com/abdullah-qasim-dev/mlops-project-23L-2640

## Requirements

The required Python dependencies are listed in `requirements.txt`.

Main libraries include:

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib

## Author

**Student ID:** 23L-2640
