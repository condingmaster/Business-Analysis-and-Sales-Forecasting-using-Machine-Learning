# Business Analysis and Sales Forecasting using Machine Learning

A complete **Data Analytics and Machine Learning project** focused on analyzing online retail transaction data, understanding customer behavior, identifying sales trends, comparing machine learning models, and forecasting future sales.

---

## Project Overview

This project uses the **Online Retail dataset** to perform business analysis and sales forecasting using Python and machine learning.

The project covers the complete workflow from data preprocessing to machine learning-based sales prediction and Power BI integration.

### Key Areas Covered

- 🧹 Data Cleaning and Preprocessing
- 📊 Business and Sales Analysis
- 👥 Customer Segmentation
- 📅 Weekly Sales Analysis
- ⚙️ Time-Series Feature Engineering
- 🤖 Machine Learning Models
- 📏 Model Evaluation using RMSE
- 🔮 Next-Week Sales Forecasting
- 📈 Data Visualization
- 📑 Power BI Integration

---

## Project Objectives

The main objectives of this project are:

1. Clean and preprocess online retail transaction data.
2. Analyze overall revenue and sales performance.
3. Identify top-selling and top revenue-generating products.
4. Analyze sales performance by country.
5. Segment customers based on their revenue contribution.
6. Identify weekly sales trends.
7. Create time-series features for sales forecasting.
8. Train and compare multiple machine learning models.
9. Evaluate model performance using RMSE.
10. Forecast sales for the following week.
11. Export processed data for Power BI visualization.

---

## Dataset

The project uses the **Online Retail Dataset**.

The input file used by the project is:

```text
Online Retail.xlsx
```

The dataset contains retail transaction information such as:

- Invoice Number
- Stock Code
- Product Description
- Quantity
- Invoice Date
- Unit Price
- Customer ID
- Country

---

##  Data Cleaning

Several preprocessing steps are performed before analysis.

### 1. Remove Missing Customer IDs

Transactions without a `CustomerID` are removed from the dataset.

### 2. Remove Invalid Transactions

Transactions with invalid values are removed:

```text
Quantity <= 0
UnitPrice <= 0
```

This helps ensure that the sales forecasting model is trained using valid sales transactions.

### 3. Convert Data Types

The following columns are converted into appropriate data types:

```text
CustomerID → Integer
InvoiceNo → String
Country → Category
```

### 4. Clean Text Columns

Product descriptions and stock codes are:

- Converted to strings
- Stripped of unnecessary spaces
- Converted to lowercase

### 5. Create TotalPrice

A new revenue column is created:

```python
TotalPrice = Quantity × UnitPrice
```

This column is used throughout the project for revenue and sales calculations.

---

## Business Analysis

The project performs several business-oriented analyses on the cleaned dataset.

### Total Revenue

The total revenue generated from the cleaned retail transactions is calculated.

### Top Selling Products

The top 10 products are identified based on total quantity sold.

### Top Revenue-Generating Products

The top 10 products generating the highest total revenue are identified.

###  Sales by Country

Sales revenue is aggregated by country to identify the countries contributing the most revenue.

---

## Customer Segmentation

Customers are grouped based on their purchasing behavior.

For each customer, the project calculates:

- Number of purchases
- Total revenue

Customers are then divided into three segments:

| Segment | Revenue Range |
|---|---:|
| Low | < 307 |
| Medium | 307 – 673 |
| High | ≥ 674 |

This segmentation provides a simple way to understand customers based on their contribution to total revenue.

---

## Weekly Sales Analysis

For sales forecasting, transaction-level data is aggregated into **weekly sales**.

The project groups transactions by week and calculates total weekly revenue.

This creates a time-series dataset containing:

```text
Date
TotalPrice
```

The weekly data is then sorted chronologically before feature engineering and model training.

---

##  Time-Series Feature Engineering

Several features are created to help machine learning models understand historical sales patterns.

### Lag Features

Previous weeks' sales are used as predictors.

```text
Lag1 → Previous week's sales
Lag2 → Sales from two weeks ago
Lag3 → Sales from three weeks ago
```

###  Time Features

The following calendar features are created:

```text
Month
Week
```

###  Rolling Mean

A **3-week rolling mean** is calculated to represent recent sales behavior.

The tree-based models use the following features:

```text
Month
Week
Lag1
Lag2
Lag3
RollingMean
```

---

## Machine Learning Models

The project compares multiple regression-based machine learning approaches.

### 1. Linear Regression

Linear Regression is trained using:

```text
Lag1
Lag2
Lag3
```

It provides a simple baseline for sales forecasting.

---

### 2. Random Forest Regressor

A Random Forest Regressor is trained using the engineered time-series features.

The model uses:

```text
n_estimators = 300
random_state = 42
```

---

### 3. XGBoost Regressor

XGBoost is used as a more advanced gradient-boosting regression model.

The model configuration includes:

```text
n_estimators = 500
learning_rate = 0.03
max_depth = 6
subsample = 0.8
colsample_bytree = 0.8
random_state = 42
```

---

### 4. XGBoost + TimeSeriesSplit

To evaluate the forecasting approach across multiple chronological splits, the project uses:

```text
TimeSeriesSplit
n_splits = 5
```

XGBoost is trained separately across the time-series splits, and the average RMSE is calculated.

---

## Model Evaluation

The models are evaluated using **RMSE (Root Mean Squared Error)**.

### RMSE

RMSE measures the difference between actual sales and predicted sales.

A **lower RMSE indicates better prediction performance**.

The project generates a model comparison table containing:

```text
Model
RMSE
```

The four approaches compared are:

```text
Linear Regression
Random Forest
XGBoost
XGBoost + TimeSeriesSplit
```

---

## Visualizations

The project generates multiple visualizations to understand sales behavior and model performance.

### Weekly Sales Trend

A line graph showing weekly revenue over time.

### Model Comparison

A bar chart comparing the RMSE values of all four model approaches.

### Actual vs Predicted Sales

A comparison of:

- Actual sales
- Linear Regression predictions
- Random Forest predictions
- XGBoost predictions
- XGBoost + TimeSeriesSplit predictions

These visualizations help evaluate how closely the models follow actual sales patterns.

---

## Next-Week Sales Forecasting

After training the models, the project creates feature values for the upcoming week using the most recent sales information.

The following models are used to generate next-week predictions:

```text
Linear Regression
Random Forest
XGBoost
XGBoost + TimeSeriesSplit
```

The predicted sales values are displayed in the console.

---

## Power BI Integration

The project exports the processed datasets and model results as CSV files.

The following files are generated:

```text
customer_data.csv
weekly_sales.csv
model_results.csv
cleaned_data.csv
```

These files can be imported into **Microsoft Power BI** to create an interactive business analytics dashboard.

### Possible Power BI Analysis

The exported data can be used to visualize:

- Sales trends
- Revenue by country
- Top products
- Customer segments
- Weekly sales
- Model RMSE comparison
- Forecasting results

---

## Technologies Used

### Programming Language

- Python

### Python Libraries

- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- XGBoost
- OpenPyXL

### Business Intelligence

- Microsoft Power BI

### Machine Learning

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor
- TimeSeriesSplit

---

## Project Structure

```text
Business-Analysis-and-Sales-Forecasting/
│
├── Online Retail.xlsx
├── mlproject.py
│
├── customer_data.csv
├── weekly_sales.csv
├── model_results.csv
├── cleaned_data.csv
│
└── README.md
```

---

##  How to Run the Project

### Step 1 — Clone the Repository

```bash
git clone <your-repository-url>
```

Navigate into the project directory:

```bash
cd Business-Analysis-and-Sales-Forecasting
```

### Step 2 — Install Required Libraries

```bash
pip install pandas numpy matplotlib scikit-learn xgboost openpyxl
```

### Step 3 — Add the Dataset

Place the following file in the project directory:

```text
Online Retail.xlsx
```

### Step 4 — Run the Python Script

```bash
python mlproject.py
```

The script will perform the analysis, train the models, generate predictions and visualizations, and export the required CSV files.

---

## Project Workflow

```text
             Online Retail Dataset
                      │
                      ▼
              Data Cleaning
                      │
                      ▼
             Business Analysis
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
 Customer Segmentation       Sales Analysis
                                  │
                                  ▼
                         Weekly Sales Data
                                  │
                                  ▼
                      Feature Engineering
                                  │
                    ┌─────────────┴─────────────┐
                    ▼             ▼             ▼
                  Lag       Time Features   Rolling Mean
                    │             │             │
                    └─────────────┴─────────────┘
                                  │
                                  ▼
                         Machine Learning
                                  │
             ┌────────────────────┼────────────────────┐
             ▼                    ▼                    ▼
      Linear Regression    Random Forest           XGBoost
                                                       │
                                                       ▼
                                              TimeSeriesSplit
                                  │
                                  ▼
                         RMSE Model Evaluation
                                  │
                                  ▼
                       Next-Week Prediction
                                  │
                                  ▼
                           CSV Export
                                  │
                                  ▼
                           Power BI Dashboard
```

## Key Features

### Business Analytics

- Total revenue analysis
- Product performance analysis
- Country-wise sales analysis
- Customer segmentation

### Machine Learning

- Time-series feature engineering
- Multiple regression models
- RMSE-based model comparison
- TimeSeriesSplit validation
- Next-week sales forecasting

### Data Visualization

- Weekly sales trend
- Model RMSE comparison
- Actual vs predicted sales

### Business Intelligence

- CSV data export
- Power BI-ready datasets

---

## Skills Demonstrated

This project demonstrates practical skills in:

- Python
- Data Cleaning
- Data Preprocessing
- Exploratory Data Analysis
- Pandas
- NumPy
- Data Visualization
- Customer Segmentation
- Time-Series Analysis
- Feature Engineering
- Regression
- Random Forest
- XGBoost
- Model Evaluation
- RMSE
- Time-Series Cross-Validation
- Sales Forecasting
- Power BI
- Business Analytics

## Project Outcome

The project provides an end-to-end workflow for transforming raw retail transaction data into useful business insights and machine learning-based sales forecasts.

It combines **business analytics, customer analysis, time-series forecasting, machine learning, and Power BI** into a single project.


## Author

**Shaurya Bisht**

B.Tech | Data Analytics & Machine Learning
