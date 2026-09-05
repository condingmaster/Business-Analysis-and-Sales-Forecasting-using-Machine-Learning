import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from xgboost import XGBRegressor

# 1. LOAD DATA

df = pd.read_excel("Online Retail.xlsx")

# 2. DATA CLEANING

# Remove missing CustomerID
df = df.dropna(subset=['CustomerID'])

# Remove invalid Quantity and UnitPrice
df = df[
    (df['Quantity'] > 0) &
    (df['UnitPrice'] > 0)
]

# Convert datatypes
df['CustomerID'] = df['CustomerID'].astype(int)

df['InvoiceNo'] = df['InvoiceNo'].astype(str)

df['Country'] = df['Country'].astype('category')

# Clean text columns
df['Description'] = (
    df['Description']
    .astype(str)
    .str.strip()
    .str.lower()
)

df['StockCode'] = (
    df['StockCode']
    .astype(str)
    .str.strip()
    .str.lower()
)

# Create TotalPrice
df['TotalPrice'] = (
    df['Quantity'] * df['UnitPrice']
)

# 3. TIME FEATURES

df['InvoiceDate'] = pd.to_datetime(
    df['InvoiceDate']
)

df['Month'] = df['InvoiceDate'].dt.month

df['Year'] = df['InvoiceDate'].dt.year

# 4. BASIC BUSINESS ANALYSIS

# Total Revenue
total_revenue = df['TotalPrice'].sum()

print("\n========== TOTAL REVENUE ==========\n")

print(total_revenue)

# Top Selling Products
top_products = (
    df.groupby('Description')['Quantity']
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\n========== TOP SELLING PRODUCTS ==========\n")

print(top_products)

# Top Revenue Products
top_revenue = (
    df.groupby('Description')['TotalPrice']
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\n========== TOP REVENUE PRODUCTS ==========\n")

print(top_revenue)

# Sales by Country
sales_by_country = (
    df.groupby('Country')['TotalPrice']
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\n========== SALES BY COUNTRY ==========\n")

print(sales_by_country)

# 5. CUSTOMER ANALYSIS

customer_data = df.groupby(
    'CustomerID'
).agg({

    'InvoiceNo': 'nunique',

    'TotalPrice': 'sum'

}).reset_index()

customer_data.columns = [

    'CustomerID',

    'NumPurchases',

    'TotalRevenue'
]

# Threshold based segmentation
conditions = [

    customer_data['TotalRevenue'] < 307,

    (
        (customer_data['TotalRevenue'] >= 307) &
        (customer_data['TotalRevenue'] < 674)
    ),

    customer_data['TotalRevenue'] >= 674
]

choices = [
    'Low',
    'Medium',
    'High'
]

customer_data['Segment'] = np.select(
    conditions,
    choices,
    default='Low'
)

print("\n========== CUSTOMER SEGMENTS ==========\n")

print(
    customer_data['Segment'].value_counts()
)

# 6. WEEKLY SALES DATA

weekly_sales = df.groupby(
    pd.Grouper(
        key='InvoiceDate',
        freq='W'
    )
)['TotalPrice'].sum().reset_index()

weekly_sales.columns = [
    'Date',
    'TotalPrice'
]

# Sort by Date
weekly_sales = weekly_sales.sort_values(
    'Date'
)

# 7. FEATURE ENGINEERING

# Month Feature
weekly_sales['Month'] = (
    weekly_sales['Date'].dt.month
)

# Week Feature
weekly_sales['Week'] = (
    weekly_sales['Date']
    .dt.isocalendar()
    .week
    .astype(int)
)

# Lag Features
weekly_sales['Lag1'] = (
    weekly_sales['TotalPrice'].shift(1)
)

weekly_sales['Lag2'] = (
    weekly_sales['TotalPrice'].shift(2)
)

weekly_sales['Lag3'] = (
    weekly_sales['TotalPrice'].shift(3)
)

# Rolling Mean
weekly_sales['RollingMean'] = (
    weekly_sales['TotalPrice']
    .rolling(3)
    .mean()
)

# Remove null rows
weekly_sales = weekly_sales.dropna()

# Remove infinite values
weekly_sales = weekly_sales.replace(
    [np.inf, -np.inf],
    np.nan
)

weekly_sales = weekly_sales.dropna()

weekly_sales = weekly_sales.reset_index(
    drop=True
)

# 8. WEEKLY SALES GRAPH

plt.close('all')

plt.figure(figsize=(12,5))

plt.plot(
    weekly_sales['Date'],
    weekly_sales['TotalPrice'],
    marker='o',
    linewidth=2
)

plt.title("Weekly Sales Trend")

plt.xlabel("Date")

plt.ylabel("Revenue")

plt.grid(True)

plt.show()

# 9. DIFFERENT FEATURES FOR DIFFERENT MODELS

# Linear Regression Features
X_lr = weekly_sales[[

    'Lag1',

    'Lag2',

    'Lag3'
]]

# Tree Based Model Features
X_tree = weekly_sales[[

    'Month',

    'Week',

    'Lag1',

    'Lag2',

    'Lag3',

    'RollingMean'
]]

y = weekly_sales['TotalPrice']

# 10. TRAIN TEST SPLIT
split = int(len(weekly_sales) * 0.8)

# Linear Regression
X_train_lr = X_lr[:split]

X_test_lr = X_lr[split:]

# Tree Models
X_train_tree = X_tree[:split]

X_test_tree = X_tree[split:]

y_train = y[:split]

y_test = y[split:]

# 11. LINEAR REGRESSION

lr = LinearRegression()

lr.fit(
    X_train_lr,
    y_train
)

lr_pred = lr.predict(
    X_test_lr
)

lr_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        lr_pred
    )
)

print(
    "\nLinear Regression RMSE:",
    lr_rmse
)

# 12. RANDOM FOREST

rf = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

rf.fit(
    X_train_tree,
    y_train
)

rf_pred = rf.predict(
    X_test_tree
)

rf_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        rf_pred
    )
)

print(
    "Random Forest RMSE:",
    rf_rmse
)

# 13. XGBOOST

xgb = XGBRegressor(

    n_estimators=500,

    learning_rate=0.03,

    max_depth=6,

    subsample=0.8,

    colsample_bytree=0.8,

    random_state=42
)

xgb.fit(
    X_train_tree,
    y_train
)

xgb_pred = xgb.predict(
    X_test_tree
)

xgb_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        xgb_pred
    )
)

print(
    "XGBoost RMSE:",
    xgb_rmse
)

# 14. TIMESERIES SPLIT
tscv = TimeSeriesSplit(
    n_splits=5
)

rmse_scores = []

ts_predictions = []

for train_index, test_index in tscv.split(X_tree):

    X_train_ts = X_tree.iloc[train_index]

    X_test_ts = X_tree.iloc[test_index]

    y_train_ts = y.iloc[train_index]

    y_test_ts = y.iloc[test_index]

    ts_model = XGBRegressor(

        n_estimators=500,

        learning_rate=0.03,

        max_depth=6,

        subsample=0.8,

        colsample_bytree=0.8,

        random_state=42
    )

    ts_model.fit(
        X_train_ts,
        y_train_ts
    )

    ts_pred = ts_model.predict(
        X_test_ts
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test_ts,
            ts_pred
        )
    )

    rmse_scores.append(rmse)

    ts_predictions.extend(ts_pred)

timeseries_rmse = np.mean(
    rmse_scores
)

print(
    "XGBoost + TimeSeriesSplit RMSE:",
    timeseries_rmse
)

# 15. MODEL COMPARISON TABLE

results_df = pd.DataFrame({

    'Model': [

        'Linear Regression',

        'Random Forest',

        'XGBoost',

        'XGBoost + TimeSeriesSplit'
    ],

    'RMSE': [

        lr_rmse,

        rf_rmse,

        xgb_rmse,

        timeseries_rmse
    ]
})

print(
    "\n========== MODEL COMPARISON ==========\n"
)

print(results_df)

# 16. RMSE COMPARISON GRAPH

models = [
    'Linear Regression',
    'Random Forest',
    'XGBoost',
    'XGBoost + TimeSeriesSplit'
]

rmse_values = [

    abs(float(lr_rmse)),

    abs(float(rf_rmse)),

    abs(float(xgb_rmse)),

    abs(float(timeseries_rmse))
]

plt.close('all')

plt.figure(figsize=(10,5))

bars = plt.bar(
    models,
    rmse_values
)

plt.title("Model Comparison (RMSE)")

plt.ylabel("RMSE")

plt.xticks(rotation=15)

# Add labels
for bar in bars:

    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f'{height:,.0f}',
        ha='center',
        va='bottom'
    )

plt.grid(axis='y')

plt.show()

# 17. ACTUAL VS PREDICTED GRAPH

plt.close('all')

plt.figure(figsize=(14,6))

plt.plot(
    y_test.values,
    label="Actual",
    marker='o',
    linewidth=3
)

plt.plot(
    lr_pred,
    label="Linear Regression",
    marker='o'
)

plt.plot(
    rf_pred,
    label="Random Forest",
    marker='o'
)

plt.plot(
    xgb_pred,
    label="XGBoost",
    marker='o',
    linestyle='--'
)

# Match same test length
plt.plot(
    ts_predictions[-len(y_test):],
    label="XGBoost + TimeSeriesSplit",
    marker='o'
)

plt.legend()

plt.title("Actual vs Predicted Sales")

plt.xlabel("Test Data")

plt.ylabel("Sales")

plt.grid(True)

plt.show()


# 18. FUTURE PREDICTION
future_features_tree = pd.DataFrame({

    'Month': [
        weekly_sales['Month'].iloc[-1]
    ],

    'Week': [
        weekly_sales['Week'].iloc[-1] + 1
    ],

    'Lag1': [
        weekly_sales['TotalPrice'].iloc[-1]
    ],

    'Lag2': [
        weekly_sales['Lag1'].iloc[-1]
    ],

    'Lag3': [
        weekly_sales['Lag2'].iloc[-1]
    ],

    'RollingMean': [
        weekly_sales['RollingMean'].iloc[-1]
    ]
})

future_features_lr = pd.DataFrame({

    'Lag1': [
        weekly_sales['TotalPrice'].iloc[-1]
    ],

    'Lag2': [
        weekly_sales['Lag1'].iloc[-1]
    ],

    'Lag3': [
        weekly_sales['Lag2'].iloc[-1]
    ]
})

prediction_lr = lr.predict(
    future_features_lr
)[0]

prediction_rf = rf.predict(
    future_features_tree
)[0]

prediction_xgb = xgb.predict(
    future_features_tree
)[0]

prediction_ts = ts_model.predict(
    future_features_tree
)[0]

print("\n========== NEXT WEEK PREDICTIONS ==========\n")

print("Linear Regression:", prediction_lr)

print("Random Forest:", prediction_rf)

print("XGBoost:", prediction_xgb)

print("XGBoost + TimeSeriesSplit:", prediction_ts)

# 19. EXPORT FILES FOR POWER BI

customer_data.to_csv(
    "customer_data.csv",
    index=False
)

weekly_sales.to_csv(
    "weekly_sales.csv",
    index=False
)

results_df.to_csv(
    "model_results.csv",
    index=False
)

df.to_csv(
    "cleaned_data.csv",
    index=False
)

print(
    "\n========== FILES EXPORTED ==========\n"
)

print("customer_data.csv exported")

print("weekly_sales.csv exported")

print("model_results.csv exported")

print("cleaned_data.csv exported")