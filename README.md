# Retail Demand Forecaster

A time-series machine learning project for forecasting daily retail product demand using historical sales patterns.

The project uses the M5 retail forecasting dataset and builds an end-to-end forecasting pipeline including exploratory analysis, feature engineering, machine learning, recursive multi-step forecasting, validation, automated testing, and an interactive Streamlit dashboard.

---

## Project Overview

Retail demand forecasting helps businesses estimate future product demand for inventory planning, replenishment, and operational decision-making.

This project focuses on forecasting demand for the top 20 products in the `CA_1` store using historical daily sales.

The forecasting pipeline predicts the next 28 days of demand.

### Key objectives

- Analyze historical retail demand
- Identify weekly and temporal demand patterns
- Build a seasonal-naive baseline
- Engineer time-series forecasting features
- Train a gradient boosting regression model
- Generate recursive 28-day forecasts
- Evaluate model performance on historical validation windows
- Build an interactive forecasting dashboard
- Add automated tests for the forecasting pipeline

---

## Dataset

The project uses the M5 Forecasting — Accuracy dataset.

The dataset contains Walmart retail unit sales across multiple stores, products, departments, categories, and states.

The main files used are:

- `sales_train_validation.csv`
- `calendar.csv`
- `sell_prices.csv`

The raw dataset is intentionally excluded from this repository because of its large size.

Download the M5 dataset separately and place the CSV files inside:

```text
data/

Project Structure
retail-demand-forecaster/
│
├── data/
│   ├── calendar.csv
│   ├── sales_train_validation.csv
│   └── sell_prices.csv
│
├── dashboard/
│   └── app.py
│
├── models/
│   └── demand_forecaster.joblib
│
├── notebooks/
│   └── 01_demand_analysis.ipynb
│
├── reports/
│   ├── forecast_28_days.csv
│   └── model_evaluation.csv
│
├── src/
│   ├── __init__.py
│   ├── evaluate.py
│   ├── features.py
│   ├── forecast.py
│   └── train.py
│
├── tests/
│   ├── test_features.py
│   └── test_forecast.py
│
├── .gitignore
├── requirements.txt
├── requirements-dev.txt
└── README.md
Modeling Approach
1. Product Selection

To keep the project computationally manageable while maintaining a realistic business scenario, the model focuses on the top 20 products by historical unit sales in store CA_1.

2. Time-Series Features

The model uses historical demand and calendar features.

Lag features
1-day lag
7-day lag
28-day lag
Rolling statistics
7-day rolling mean
28-day rolling mean
28-day rolling standard deviation
Calendar features
Day of week
Month
Week of year
Weekend indicator

These features allow the model to capture short-term demand, weekly seasonality, longer-term patterns, and temporal effects.

Baseline

A seasonal-naive forecasting approach is used as the baseline.

The baseline predicts demand using sales from the same weekday one week earlier.

Historical validation produced approximately:

Metric	Seasonal Naive
MAE	8.15
RMSE	13.34

The baseline provides a simple benchmark for determining whether the machine learning model adds predictive value.

Machine Learning Model

The main model is:

HistGradientBoostingRegressor

Configuration:

max_iter = 300
learning_rate = 0.05
max_leaf_nodes = 31
l2_regularization = 1.0
random_state = 42

The model is trained using historical observations up to the selected training cutoff.

Recursive Forecasting

The production forecast uses recursive multi-step prediction.

For each future day:

Generate features from historical observations
Predict demand
Add the prediction to the history
Use the prediction when generating features for the next day
Repeat for the full 28-day horizon

This avoids using actual future sales during forecasting.

The final production forecast covers:

2016-04-25 → 2016-05-22

for 20 products, producing:

20 × 28 = 560

forecast observations.

Validation

The model is evaluated using three historical 28-day validation windows.

Validation Window	MAE	RMSE
2016-02-01 → 2016-02-28	7.75	11.38
2016-02-29 → 2016-03-27	8.09	11.26
2016-03-28 → 2016-04-24	7.76	10.71
Average	7.87	11.12

The validation forecasts are generated recursively.

Important validation note

The current evaluation uses the same trained model across the three historical validation windows.

Therefore, this should be interpreted as rolling historical evaluation rather than strict walk-forward retraining at every fold.

Results

The machine learning model achieved:

MAE  = 7.87
RMSE = 11.12

compared with the seasonal-naive baseline:

MAE  = 8.15
RMSE = 13.34

This indicates that the engineered machine-learning approach improves the historical validation error relative to the simple seasonal baseline.

The model also shows some positive forecast bias in total demand on the validation windows, particularly in the later periods. This is an area for future improvement.

Forecast Output

The production pipeline generated a 28-day forecast for the selected 20 products.

Total forecasted demand:

~12,102 units

The forecast is stored in:

reports/forecast_28_days.csv
Dashboard

The project includes an interactive Streamlit dashboard.

The dashboard contains three views:

Product Forecast

View:

Historical demand
28-day forecast
Daily predictions
Product-level forecast statistics
Portfolio Overview

View:

Total forecasted units
Daily portfolio demand
Product forecast comparison
Product-level summary
Model Performance

View:

MAE
RMSE
Validation performance
Actual vs predicted demand
Validation fold results
Running the Project
1. Clone the repository
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd retail-demand-forecaster
2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate
3. Install dependencies
pip install -r requirements.txt

For development and testing:

pip install -r requirements-dev.txt
4. Add the dataset

Place:

calendar.csv
sales_train_validation.csv
sell_prices.csv

inside:

data/
Train the Model

Run:

PYTHONPATH=. python -m src.train

This creates:

models/demand_forecaster.joblib
Generate the 28-Day Forecast

Run:

PYTHONPATH=. python -m src.forecast

The forecast is saved to:

reports/forecast_28_days.csv
Evaluate the Model

Run:

PYTHONPATH=. python -m src.evaluate

Evaluation results are saved to:

reports/model_evaluation.csv
Run Tests

Run:

PYTHONPATH=. pytest tests/ -v

The project currently includes tests covering:

Feature creation
Lag features
Missing feature values
Forecast row counts
Forecast product coverage
Forecast dates
Non-negative predictions
Launch the Dashboard

Run:

PYTHONPATH=. streamlit run dashboard/app.py

Then open the local Streamlit URL displayed in the terminal.

Key Engineering Decisions
Why not a random train/test split?

Random splitting can place observations from the future into the training set while evaluating on earlier observations.

For forecasting, the model should only learn from information that would have been available at prediction time.

Therefore, this project uses chronological validation windows.

Why recursive forecasting?

A 28-day forecast requires predictions for future dates where actual sales are unavailable.

Recursive forecasting allows the model to generate the first prediction and then use that prediction as part of the input for subsequent days.

Why use a baseline?

A machine learning model should be compared against a simple forecasting strategy.

The seasonal-naive model establishes whether the additional model complexity provides useful improvement.

Limitations

This project intentionally uses a reduced subset of the M5 dataset.

Current limitations include:

Only one store is modeled
Only the top 20 products are modeled
The model does not use the full product-store hierarchy
The current validation does not retrain the model at every fold
The original M5 WRMSSE competition metric is not implemented
Price and event features were investigated but did not materially improve the selected model
Forecast uncertainty intervals are not currently generated
Future Improvements

Potential extensions include:

Strict walk-forward retraining
Modeling all store-product combinations
Hierarchical forecasting
WRMSSE evaluation
Price elasticity features
Promotion/event interactions
More advanced lag features
LightGBM or XGBoost comparison
Probabilistic forecasting
Prediction intervals
Automated model retraining
CI/CD testing
Cloud deployment
Technologies
Python
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
Plotly
Streamlit
Joblib
Pytest
Jupyter
Project Highlights

This project demonstrates:

Time-series feature engineering
Forecasting-specific validation
Recursive multi-step prediction
Gradient boosting regression
Baseline comparison
Model evaluation
Automated testing
Production-style Python modules
Interactive data visualization
Reproducible ML workflows
Author

CodingAgent006