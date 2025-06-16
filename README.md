# 🏎️ Formula 1 Winner Prediction

Predict the winner of the 2025 Austrian Grand Prix using historical F1 data and machine learning.

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Usage](#usage)
6. [API Documentation](#api-documentation)
7. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
8. [Model Training](#model-training)
9. [Testing](#testing)
10. [Contributing](#contributing)
11. [License](#license)

---

## Overview
This project uses the `fastf1` library to fetch historical Formula 1 data, preprocess it, and train a machine learning model to predict the winner of the 2025 Austrian Grand Prix. The project includes a Flask API for real-time predictions and Jupyter Notebooks for experimentation and EDA.

---

## Features
- **Data Collection**: Fetch race results and lap data using the `fastf1` library.
- **Preprocessing**: Clean and engineer features from raw data.
- **Exploratory Data Analysis (EDA)**: Visualize trends and correlations in the data.
- **Machine Learning Model**: Train a Random Forest classifier to predict winners.
- **Flask API**: Deploy the model for real-time predictions.
- **CI/CD Pipeline**: Automate testing and deployment using GitHub Actions.
- **Modular Design**: Organized codebase with reusable modules.

---

## Project Structure
The project is organized as follows:

f1_winner_prediction/
│
├── README.md # Project overview, setup instructions, and usage
├── requirements.txt # List of required Python libraries
├── data/ # Folder for raw and processed data
│ ├── raw_data/ # Raw data fetched from APIs or fastf1
│ └── processed_data/ # Preprocessed data files (e.g., CSVs)
│
├── src/ # Source code for the project
│ ├── init .py # Makes src a Python package
│ ├── data_collection.py # Module for fetching and saving data using fastf1
│ ├── preprocessing.py # Module for data cleaning and feature engineering
│ ├── eda.py # Exploratory Data Analysis (EDA) scripts
│ ├── modeling.py # Model training and evaluation
│ ├── api.py # Flask API for real-time predictions
│ └── utils.py # Utility functions (e.g., logging, helpers)
│
├── notebooks/ # Jupyter Notebooks for experimentation and EDA
│ ├── data_exploration.ipynb # Notebook for initial data exploration
│ ├── feature_engineering.ipynb # Notebook for feature engineering
│ └── model_training.ipynb # Notebook for model experimentation
│
├── models/ # Trained models (saved as .pkl or .joblib)
│ ├── random_forest_model.pkl # Example: Saved Random Forest model
│
├── logs/ # Logs for debugging and monitoring
│ ├── app.log # Log file for the Flask API
│
├── tests/ # Unit tests for the project
│ ├── test_data_collection.py # Tests for data collection module
│ ├── test_preprocessing.py # Tests for preprocessing module
│ └── test_api.py # Tests for the Flask API
│
└── .github/ # GitHub Actions workflows
└── workflows/ # CI/CD pipelines
└── ci-cd.yml # Example: Workflow for automated testing and deployment


---

## Installation
### Prerequisites
- Python 3.9 or higher
- Git

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/mhdbel/f1_winner_prediction.git 
   cd f1_winner_prediction

2. Install dependencies:
    pip install -r requirements.txt

3. Enable caching for fastf1 (optional but recommended):
    import fastf1
    fastf1.Cache.enable_cache('cache')

4. Acknowledgements:
    Thanks to the developers of the fastf1 library for providing access to F1 data.
    Inspired by Formula 1 enthusiasts and data science communities.

    Copyright (c) 2025 mhdbel
    
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    In the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    
    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS ARE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OF OTHER DEALINGS IN THE
    SOFTWARE.
