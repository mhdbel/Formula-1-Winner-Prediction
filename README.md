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
├── README.md
├── requirements.txt
├── data/
│ ├── raw_data/
│ └── processed_data/
│
├── src/
│ ├── init .py
│ ├── data_collection.py
│ ├── preprocessing.py
│ ├── eda.py
│ ├── modeling.py
│ ├── api.py
│ └── utils.py
│
├── notebooks/
│ ├── data_exploration.ipynb
│ ├── feature_engineering.ipynb
│ └── model_training.ipynb
│
├── models/
│ ├── random_forest_model.pkl
│
├── logs/
│ ├── app.log
│
├── tests/
│ ├── test_data_collection.py
│ ├── test_preprocessing.py
│ └── test_api.py #
│
└── .github/
└── workflows/
└── ci-cd.yml


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
