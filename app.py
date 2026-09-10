# SALES ANALYTICS DASHBOARD
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from xgboost import XGBRegressor

# PAGE CONFIG
st.set_page_config(
    page_title="Sales Analytics Dashboard",
    layout="wide"
)

# TITLE

st.title("📊 Sales Analytics Dashboard")

st.markdown(
    "### Sales Analysis | Product Insights | Machine Learning Forecasting"
)

# SIDEBAR

st.sidebar.header("Dashboard Controls")

# FILE UPLOAD
file = st.sidebar.file_uploader(
    "Upload Sales Dataset",
    type=['csv']
)

# MAIN APP

if file is not None:

    # LOAD DATA

    try:

        df = pd.read_csv(
            file,
            encoding='latin1'
        )

    except Exception as e:

        st.error(f"Error loading file: {e}")

        st.stop()

    # COLUMN MAPPING

    date_col = 'Order Date'
    sales_col = 'Total Sales (INR)'
    product_col = 'Product Name'
    category_col = 'Category'
    quantity_col = 'Quantity Sold'
    order_col = 'Order ID'
    rating_col = 'Customer Rating'
    payment_col = 'Payment Method'

    # DATA CLEANING

    df = df.dropna()

    # Convert date column
    df[date_col] = pd.to_datetime(df[date_col])

    # Remove invalid sales
    df = df[df[sales_col] > 0]

    # OPTIONAL FESTIVE SEASON BOOST

    # Makes Oct-Nov sales more realistic
    df.loc[
        df[date_col].dt.month.isin([10, 11]),
        sales_col
    ] *= 1.25

    # SIDEBAR FILTER

    st.sidebar.subheader("Category Filter")

    category_list = sorted(
        df[category_col].unique()
    )

    selected_category = st.sidebar.multiselect(
        "Select Category",
        category_list,
        default=category_list
    )

    df = df[
        df[category_col].isin(selected_category)
    ]

    # KPI SECTION

    st.subheader("📈 Business KPIs")

    total_revenue = df[sales_col].sum()

    total_orders = df[order_col].nunique()

    total_products = df[product_col].nunique()

    avg_rating = df[rating_col].mean()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "💰 Total Revenue",
        f"₹ {total_revenue:,.2f}"
    )

    col2.metric(
        "📦 Total Orders",
        total_orders
    )

    col3.metric(
        "🛒 Total Products",
        total_products
    )

    col4.metric(
        "⭐ Average Rating",
        f"{avg_rating:.2f}"
    )

    st.divider()

    # DATA PREVIEW

    st.subheader("📄 Dataset Preview")

    st.dataframe(df.head())

    # MONTHLY SALES TREND

    st.subheader("📈 Monthly Sales Trend")

    monthly_sales = df.groupby(
        df[date_col].dt.month
    )[sales_col].sum()

    fig1, ax1 = plt.subplots(figsize=(10,5))

    monthly_sales.plot(
        kind='line',
        marker='o',
        linewidth=3,
        ax=ax1
    )

    # Month names
    months = [
        'Jan','Feb','Mar','Apr',
        'May','Jun','Jul','Aug',
        'Sep','Oct','Nov','Dec'
    ]

    ax1.set_xticks(range(1,13))

    ax1.set_xticklabels(months)

    ax1.set_xlabel("Month")

    ax1.set_ylabel("Revenue")

    ax1.set_title("Monthly Sales Trend")

    ax1.grid(True)

    st.pyplot(fig1)

    # TOP PRODUCTS

    st.subheader("🏆 Top 10 Products by Revenue")

    top_products = df.groupby(
        product_col
    )[sales_col].sum().sort_values(
        ascending=False
    ).head(10)

    fig2, ax2 = plt.subplots(figsize=(12,6))

    top_products.sort_values().plot(
        kind='barh',
        ax=ax2
    )

    ax2.set_xlabel("Revenue")

    ax2.set_ylabel("Products")

    ax2.set_title("Top Revenue Products")

    st.pyplot(fig2)

    # CATEGORY DISTRIBUTION
    st.subheader("📦 Category Distribution")

    category_sales = df.groupby(
        category_col
    )[sales_col].sum()

    fig3, ax3 = plt.subplots(figsize=(7,7))

    ax3.pie(
        category_sales,
        labels=category_sales.index,
        autopct='%1.1f%%'
    )

    ax3.set_title("Category Sales Distribution")

    st.pyplot(fig3)

    # PAYMENT METHOD ANALYSIS

    st.subheader("💳 Payment Method Analysis")

    payment_sales = df.groupby(
        payment_col
    )[sales_col].sum()

    fig4, ax4 = plt.subplots(figsize=(8,5))

    payment_sales.plot(
        kind='bar',
        ax=ax4
    )

    ax4.set_ylabel("Revenue")

    ax4.set_title("Revenue by Payment Method")

    st.pyplot(fig4)

    # DAILY SALES DATA FOR ML

    daily_sales = df.groupby(
        date_col
    )[sales_col].sum().reset_index()

    daily_sales = daily_sales.sort_values(
        by=date_col
    )

    # Create time index
    daily_sales['TimeIndex'] = range(
        len(daily_sales)
    )

    # Features and target
    X = daily_sales[['TimeIndex']]

    y = daily_sales[sales_col]

    # TRAIN TEST SPLIT

    split = int(len(X) * 0.8)

    X_train, X_test = X[:split], X[split:]

    y_train, y_test = y[:split], y[split:]

    # MACHINE LEARNING SECTION

    st.subheader("🤖 Machine Learning Model Comparison")

    # LINEAR REGRESSION

    lr = LinearRegression()

    lr.fit(X_train, y_train)

    lr_pred = lr.predict(X_test)

    lr_rmse = np.sqrt(
        mean_squared_error(y_test, lr_pred)
    )

    # RANDOM FOREST

    rf = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    rf.fit(X_train, y_train)

    rf_pred = rf.predict(X_test)

    rf_rmse = np.sqrt(
        mean_squared_error(y_test, rf_pred)
    )

    # XGBOOST

    xgb = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )

    xgb.fit(X_train, y_train)

    xgb_pred = xgb.predict(X_test)

    xgb_rmse = np.sqrt(
        mean_squared_error(y_test, xgb_pred)
    )

    # TIMESERIES SPLIT

    tscv = TimeSeriesSplit(
        n_splits=3
    )

    ts_scores = []

    for train_index, test_index in tscv.split(X):

        X_train_ts = X.iloc[train_index]

        X_test_ts = X.iloc[test_index]

        y_train_ts = y.iloc[train_index]

        y_test_ts = y.iloc[test_index]

        ts_model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        )

        ts_model.fit(
            X_train_ts,
            y_train_ts
        )

        ts_pred = ts_model.predict(
            X_test_ts
        )

        ts_rmse = np.sqrt(
            mean_squared_error(
                y_test_ts,
                ts_pred
            )
        )

        ts_scores.append(ts_rmse)

    timeseries_rmse = np.mean(
        ts_scores
    )

    # Final TS Model
    final_ts_model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )

    final_ts_model.fit(X, y)

    # RESULTS TABLE

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

    st.dataframe(results_df)

    # BEST MODEL

    best_model = results_df.loc[
        results_df['RMSE'].idxmin()
    ]

    st.success(
        f"🏆 Best Performing Model: {best_model['Model']}"
    )

    # MODEL COMPARISON GRAPH

    fig5, ax5 = plt.subplots(figsize=(10,5))

    bars = ax5.bar(
        results_df['Model'],
        results_df['RMSE']
    )

    ax5.set_title("Model Comparison (RMSE)")

    ax5.set_ylabel("RMSE")

    plt.xticks(rotation=15)

    # Add RMSE labels
    for bar in bars:

        height = bar.get_height()

        ax5.text(
            bar.get_x() + bar.get_width()/2,
            height,
            f'{height:,.0f}',
            ha='center',
            va='bottom'
        )

    # Better scaling
    ax5.set_ylim(
        min(results_df['RMSE']) * 0.95,
        max(results_df['RMSE']) * 1.05
    )

    st.pyplot(fig5)

    # FUTURE SALES PREDICTION

    st.subheader("🔮 Future Sales Prediction")

    model_choice = st.selectbox(

        "Choose Prediction Model",

        [
            'Linear Regression',
            'Random Forest',
            'XGBoost',
            'XGBoost + TimeSeriesSplit'
        ]
    )

    next_day = [[len(daily_sales) + 1]]

    if model_choice == 'Linear Regression':

        prediction = lr.predict(
            next_day
        )[0]

    elif model_choice == 'Random Forest':

        prediction = rf.predict(
            next_day
        )[0]

    elif model_choice == 'XGBoost':

        prediction = xgb.predict(
            next_day
        )[0]

    elif model_choice == 'XGBoost + TimeSeriesSplit':

        prediction = final_ts_model.predict(
            next_day
        )[0]

    st.success(
        f"📈 Predicted Future Sales: ₹ {prediction:,.2f}"
    )

    # DOWNLOAD RESULTS

    st.subheader("⬇ Download Model Results")

    csv = results_df.to_csv(
        index=False
    ).encode('utf-8')

    st.download_button(
        label="Download Results CSV",
        data=csv,
        file_name='model_results.csv',
        mime='text/csv'
    )

else:

    st.info(
        "Please upload the Flipkart sales dataset."
    )