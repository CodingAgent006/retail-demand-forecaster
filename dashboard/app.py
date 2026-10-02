from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"

SALES_PATH = DATA_DIR / "sales_train_validation.csv"
CALENDAR_PATH = DATA_DIR / "calendar.csv"

FORECAST_PATH = REPORTS_DIR / "forecast_28_days.csv"
EVALUATION_PATH = REPORTS_DIR / "model_evaluation.csv"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Retail Demand Forecaster",
    page_icon="📈",
    layout="wide",
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_forecast():
    forecast = pd.read_csv(FORECAST_PATH)
    forecast["date"] = pd.to_datetime(forecast["date"])
    return forecast


@st.cache_data
def load_sales():
    sales = pd.read_csv(SALES_PATH)
    return sales


@st.cache_data
def load_calendar():
    calendar = pd.read_csv(CALENDAR_PATH)
    calendar["date"] = pd.to_datetime(calendar["date"])
    return calendar


@st.cache_data
def load_evaluation():
    evaluation = pd.read_csv(EVALUATION_PATH)

    return evaluation


@st.cache_data
def load_historical_product(
    item_id,
    store_id="CA_1",
):
    sales = load_sales()
    calendar = load_calendar()

    product = sales[
        (sales["item_id"] == item_id)
        & (sales["store_id"] == store_id)
    ].copy()

    day_columns = [
        column
        for column in sales.columns
        if column.startswith("d_")
    ]

    product_long = product.melt(
        id_vars=[
            "item_id",
            "dept_id",
            "cat_id",
            "store_id",
        ],
        value_vars=day_columns,
        var_name="d",
        value_name="sales",
    )

    product_long = product_long.merge(
        calendar[["d", "date"]],
        on="d",
        how="left",
    )

    product_long["date"] = pd.to_datetime(
        product_long["date"]
    )

    return product_long[
        [
            "item_id",
            "store_id",
            "date",
            "sales",
        ]
    ]


# ============================================================
# LOAD DATA
# ============================================================

try:
    forecast = load_forecast()
    evaluation = load_evaluation()

except FileNotFoundError as error:
    st.error(
        f"Required project file was not found:\n\n{error}"
    )
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📈 Retail Demand Forecaster")

st.sidebar.markdown(
    """
Forecasting dashboard for the M5 retail demand dataset.

**Store:** CA_1  
**Products:** Top 20 by historical unit sales  
**Forecast horizon:** 28 days
"""
)

page = st.sidebar.radio(
    "Dashboard View",
    [
        "Product Forecast",
        "Portfolio Overview",
        "Model Performance",
    ],
)


# ============================================================
# TITLE
# ============================================================

st.title("Retail Demand Forecaster")

st.caption(
    "Time-series demand forecasting using recursive "
    "machine learning and historical demand features."
)


# ============================================================
# PAGE 1 — PRODUCT FORECAST
# ============================================================

if page == "Product Forecast":

    st.header("Product Forecast")

    products = sorted(
        forecast["item_id"].unique()
    )

    selected_product = st.selectbox(
        "Select a product",
        products,
    )

    product_forecast = forecast[
        forecast["item_id"] == selected_product
    ].copy()

    historical = load_historical_product(
        selected_product
    )

    # Keep recent history for a readable chart.
    history_start = (
        product_forecast["date"].min()
        - pd.Timedelta(days=56)
    )

    historical_recent = historical[
        historical["date"] >= history_start
    ].copy()

    # --------------------------------------------------------
    # Product information
    # --------------------------------------------------------

    product_info = forecast[
        forecast["item_id"] == selected_product
    ].iloc[0]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Forecast Units",
            f"{product_forecast['prediction'].sum():,.0f}",
        )

    with col2:
        st.metric(
            "Daily Average",
            f"{product_forecast['prediction'].mean():.1f}",
        )

    with col3:
        st.metric(
            "Peak Forecast",
            f"{product_forecast['prediction'].max():.1f}",
        )

    with col4:
        st.metric(
            "Forecast Days",
            f"{product_forecast['date'].nunique()}",
        )

    st.divider()

    # --------------------------------------------------------
    # Historical + forecast chart
    # --------------------------------------------------------

    historical_plot = historical_recent[
        ["date", "sales"]
    ].copy()

    historical_plot["series"] = "Historical"
    historical_plot = historical_plot.rename(
        columns={"sales": "units"}
    )

    forecast_plot = product_forecast[
        ["date", "prediction"]
    ].copy()

    forecast_plot["series"] = "Forecast"
    forecast_plot = forecast_plot.rename(
        columns={"prediction": "units"}
    )

    chart_data = pd.concat(
        [
            historical_plot,
            forecast_plot,
        ],
        ignore_index=True,
    )

    fig = px.line(
        chart_data,
        x="date",
        y="units",
        color="series",
        markers=True,
        title=f"{selected_product} — Historical Demand and 28-Day Forecast",
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Units",
        hovermode="x unified",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # Daily forecast table
    # --------------------------------------------------------

    st.subheader("Daily Forecast")

    display_forecast = product_forecast[
        ["date", "prediction"]
    ].copy()

    display_forecast["date"] = display_forecast[
        "date"
    ].dt.strftime("%Y-%m-%d")

    display_forecast["prediction"] = (
        display_forecast["prediction"]
        .round(2)
    )

    display_forecast = display_forecast.rename(
        columns={
            "date": "Date",
            "prediction": "Forecast Units",
        }
    )

    st.dataframe(
        display_forecast,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# PAGE 2 — PORTFOLIO OVERVIEW
# ============================================================

elif page == "Portfolio Overview":

    st.header("Portfolio Overview")

    # --------------------------------------------------------
    # Portfolio metrics
    # --------------------------------------------------------

    total_forecast = forecast["prediction"].sum()

    average_daily_forecast = (
        forecast.groupby("date")["prediction"]
        .sum()
        .mean()
    )

    peak_day = (
        forecast.groupby("date")["prediction"]
        .sum()
        .idxmax()
    )

    peak_day_units = (
        forecast.groupby("date")["prediction"]
        .sum()
        .max()
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Forecast",
            f"{total_forecast:,.0f}",
        )

    with col2:
        st.metric(
            "Average Daily Units",
            f"{average_daily_forecast:,.0f}",
        )

    with col3:
        st.metric(
            "Products",
            f"{forecast['item_id'].nunique()}",
        )

    with col4:
        st.metric(
            "Peak Daily Demand",
            f"{peak_day_units:,.0f}",
        )

    st.divider()

    # --------------------------------------------------------
    # Daily portfolio forecast
    # --------------------------------------------------------

    st.subheader("Portfolio Demand by Day")

    daily_forecast = (
        forecast.groupby("date", as_index=False)
        ["prediction"]
        .sum()
    )

    fig_daily = px.line(
        daily_forecast,
        x="date",
        y="prediction",
        markers=True,
        title="Total Forecasted Units by Day",
    )

    fig_daily.update_layout(
        xaxis_title="Date",
        yaxis_title="Forecasted Units",
        hovermode="x unified",
    )

    st.plotly_chart(
        fig_daily,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # Product ranking
    # --------------------------------------------------------

    st.subheader("Forecast by Product")

    product_summary = (
        forecast.groupby("item_id", as_index=False)
        ["prediction"]
        .sum()
        .sort_values(
            "prediction",
            ascending=False,
        )
    )

    fig_products = px.bar(
        product_summary,
        x="item_id",
        y="prediction",
        title="28-Day Forecast by Product",
    )

    fig_products.update_layout(
        xaxis_title="Product",
        yaxis_title="Forecasted Units",
    )

    st.plotly_chart(
        fig_products,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # Summary table
    # --------------------------------------------------------

    st.subheader("Product Forecast Summary")

    summary = (
        forecast.groupby("item_id")
        .agg(
            total_forecast=("prediction", "sum"),
            average_daily=("prediction", "mean"),
            peak_daily=("prediction", "max"),
        )
        .reset_index()
        .sort_values(
            "total_forecast",
            ascending=False,
        )
    )

    summary["total_forecast"] = (
        summary["total_forecast"]
        .round(0)
        .astype(int)
    )

    summary["average_daily"] = (
        summary["average_daily"]
        .round(2)
    )

    summary["peak_daily"] = (
        summary["peak_daily"]
        .round(2)
    )

    summary = summary.rename(
        columns={
            "item_id": "Product",
            "total_forecast": "28-Day Forecast",
            "average_daily": "Average Daily",
            "peak_daily": "Peak Daily",
        }
    )

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# PAGE 3 — MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.header("Model Performance")

    st.markdown(
        """
The model was evaluated using historical validation windows.
The evaluation uses recursive forecasting, meaning predictions
are fed back into the model during the forecast horizon.
"""
    )

    # --------------------------------------------------------
    # Read average model metrics
    # --------------------------------------------------------

    average_results = evaluation[
        evaluation["validation_start"] == "AVERAGE"
    ]

    if average_results.empty:
        st.warning(
            "AVERAGE row was not found in "
            "model_evaluation.csv."
        )
        st.stop()

    average_results = average_results.iloc[0]

    model_mae = float(
        average_results["mae"]
    )

    model_rmse = float(
        average_results["rmse"]
    )

    # Baseline values from the seasonal-naive benchmark.
    baseline_mae = 8.15
    baseline_rmse = 13.34

    mae_improvement = (
        (baseline_mae - model_mae)
        / baseline_mae
        * 100
    )

    rmse_improvement = (
        (baseline_rmse - model_rmse)
        / baseline_rmse
        * 100
    )

    # --------------------------------------------------------
    # Metric cards
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Model MAE",
            f"{model_mae:.2f}",
        )

    with col2:
        st.metric(
            "Baseline MAE",
            f"{baseline_mae:.2f}",
            delta=f"{mae_improvement:.1f}%",
        )

    with col3:
        st.metric(
            "Model RMSE",
            f"{model_rmse:.2f}",
        )

    with col4:
        st.metric(
            "Baseline RMSE",
            f"{baseline_rmse:.2f}",
            delta=f"{rmse_improvement:.1f}%",
        )

    st.divider()

    # --------------------------------------------------------
    # Validation results
    # --------------------------------------------------------

    st.subheader("Validation Results")

    fold_results = evaluation[
        evaluation["validation_start"] != "AVERAGE"
    ].copy()

    fold_results["fold"] = [
        "Fold 1",
        "Fold 2",
        "Fold 3",
    ][:len(fold_results)]

    fig_metrics = px.bar(
        fold_results,
        x="fold",
        y=["mae", "rmse"],
        barmode="group",
        title="Model Error Across Validation Windows",
    )

    fig_metrics.update_layout(
        xaxis_title="Validation Fold",
        yaxis_title="Error",
        legend_title="Metric",
    )

    st.plotly_chart(
        fig_metrics,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # Actual vs predicted totals
    # --------------------------------------------------------

    st.subheader(
        "Actual vs Predicted Demand by Validation Window"
    )

    comparison = fold_results[
        [
            "fold",
            "actual_total",
            "predicted_total",
        ]
    ].copy()

    comparison_long = comparison.melt(
        id_vars="fold",
        value_vars=[
            "actual_total",
            "predicted_total",
        ],
        var_name="type",
        value_name="units",
    )

    comparison_long["type"] = (
        comparison_long["type"]
        .map(
            {
                "actual_total": "Actual",
                "predicted_total": "Predicted",
            }
        )
    )

    fig_totals = px.bar(
        comparison_long,
        x="fold",
        y="units",
        color="type",
        barmode="group",
        title="Actual vs Predicted Units",
    )

    fig_totals.update_layout(
        xaxis_title="Validation Fold",
        yaxis_title="Units",
    )

    st.plotly_chart(
        fig_totals,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # Detailed results table
    # --------------------------------------------------------

    st.subheader("Detailed Evaluation")

    display_results = evaluation.copy()

    display_results["mae"] = (
        display_results["mae"]
        .round(4)
    )

    display_results["rmse"] = (
        display_results["rmse"]
        .round(4)
    )

    display_results["actual_total"] = (
        display_results["actual_total"]
        .round(0)
    )

    display_results["predicted_total"] = (
        display_results["predicted_total"]
        .round(2)
    )

    display_results = display_results.rename(
        columns={
            "validation_start": "Validation Start",
            "validation_end": "Validation End",
            "mae": "MAE",
            "rmse": "RMSE",
            "actual_total": "Actual Units",
            "predicted_total": "Predicted Units",
        }
    )

    st.dataframe(
        display_results,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # Model explanation
    # --------------------------------------------------------

    st.subheader("Model Configuration")

    st.markdown(
        """
**Algorithm:** HistGradientBoostingRegressor

**Demand features:**
- 1-day lag
- 7-day lag
- 28-day lag
- 7-day rolling mean
- 28-day rolling mean
- 28-day rolling standard deviation
- Day of week
- Month
- Week of year
- Weekend indicator

**Forecasting strategy:** Recursive 28-day forecasting

**Validation strategy:** Three historical validation windows

**Target:** Daily unit sales

**Store:** CA_1

**Product scope:** Top 20 products by historical sales
"""
    )

    st.info(
        "Validation metrics describe historical performance. "
        "They are not accuracy measurements for the future "
        "28-day forecast."
    )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "Retail Demand Forecaster • M5-inspired forecasting project"
)