from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

SALES_PATH = DATA_DIR / "sales_train_validation.csv"
CALENDAR_PATH = DATA_DIR / "calendar.csv"
MODEL_PATH = MODEL_DIR / "demand_forecaster.joblib"
FORECAST_PATH = REPORT_DIR / "forecast_28_days.csv"


# ============================================================
# CONFIGURATION
# ============================================================

STORE_ID = "CA_1"

FORECAST_START = "2016-04-25"
FORECAST_END = "2016-05-22"

N_TOP_PRODUCTS = 20


# ============================================================
# LOAD DATA
# ============================================================

def load_sales():

    return pd.read_csv(
        SALES_PATH
    )


def load_calendar():

    calendar = pd.read_csv(
        CALENDAR_PATH
    )

    calendar["date"] = pd.to_datetime(
        calendar["date"]
    )

    return calendar


# ============================================================
# SELECT TOP PRODUCTS
# ============================================================

def select_top_products(
    sales,
    n_products=20
):

    store_sales = sales[
        sales["store_id"] == STORE_ID
    ].copy()

    day_columns = [
        column
        for column in sales.columns
        if column.startswith("d_")
    ]

    store_sales["total_units"] = (
        store_sales[day_columns]
        .sum(axis=1)
    )

    top_products = (
        store_sales
        .sort_values(
            "total_units",
            ascending=False
        )
        .head(n_products)
    )

    return top_products["item_id"].tolist()


# ============================================================
# RESHAPE SALES
# ============================================================

def reshape_sales(
    sales,
    selected_items
):

    sales = sales[
        (sales["store_id"] == STORE_ID)
        & (
            sales["item_id"].isin(
                selected_items
            )
        )
    ].copy()

    day_columns = [
        column
        for column in sales.columns
        if column.startswith("d_")
    ]

    sales_long = sales.melt(
        id_vars=[
            "item_id",
            "dept_id",
            "cat_id",
            "store_id"
        ],
        value_vars=day_columns,
        var_name="d",
        value_name="sales"
    )

    return sales_long


# ============================================================
# MERGE CALENDAR
# ============================================================

def merge_calendar(
    sales_long,
    calendar
):

    calendar_small = calendar[
        [
            "d",
            "date"
        ]
    ].copy()

    sales_long = sales_long.merge(
        calendar_small,
        on="d",
        how="left"
    )

    sales_long["date"] = pd.to_datetime(
        sales_long["date"]
    )

    return sales_long


# ============================================================
# RECURSIVE FORECAST
# ============================================================

def recursive_forecast(
    model,
    history,
    future_dates
):

    history = history[
        [
            "item_id",
            "date",
            "sales"
        ]
    ].copy()

    history["date"] = pd.to_datetime(
        history["date"]
    )

    history = history.sort_values(
        [
            "item_id",
            "date"
        ]
    )

    predictions = []

    items = history[
        "item_id"
    ].unique()

    for item_id in items:

        item_history = history[
            history["item_id"] == item_id
        ].copy()

        item_history = item_history.sort_values(
            "date"
        )

        for forecast_date in future_dates:

            sales_values = (
                item_history["sales"]
                .values
            )

            lag_1 = sales_values[-1]

            lag_7 = sales_values[-7]

            lag_28 = sales_values[-28]

            rolling_mean_7 = (
                sales_values[-7:]
                .mean()
            )

            rolling_mean_28 = (
                sales_values[-28:]
                .mean()
            )

            rolling_std_28 = (
                sales_values[-28:]
                .std()
            )

            day_of_week = (
                forecast_date.dayofweek
            )

            month = (
                forecast_date.month
            )

            week_of_year = (
                forecast_date
                .isocalendar()
                .week
            )

            is_weekend = int(
                day_of_week >= 5
            )

            feature_row = pd.DataFrame(
                [
                    {
                        "lag_1": lag_1,
                        "lag_7": lag_7,
                        "lag_28": lag_28,
                        "rolling_mean_7": rolling_mean_7,
                        "rolling_mean_28": rolling_mean_28,
                        "rolling_std_28": rolling_std_28,
                        "day_of_week": day_of_week,
                        "month": month,
                        "week_of_year": int(
                            week_of_year
                        ),
                        "is_weekend": is_weekend,
                    }
                ]
            )

            prediction = model.predict(
                feature_row
            )[0]

            # Demand cannot be negative.
            prediction = max(
                0,
                prediction
            )

            predictions.append(
                {
                    "item_id": item_id,
                    "date": forecast_date,
                    "prediction": prediction
                }
            )

            # Feed prediction back into history
            # for the next recursive step.
            item_history = pd.concat(
                [
                    item_history,
                    pd.DataFrame(
                        [
                            {
                                "item_id": item_id,
                                "date": forecast_date,
                                "sales": prediction
                            }
                        ]
                    )
                ],
                ignore_index=True
            )

    return pd.DataFrame(
        predictions
    )


# ============================================================
# MAIN FORECAST PIPELINE
# ============================================================

def generate_forecast():

    print("=" * 60)
    print("RETAIL DEMAND FORECAST")
    print("=" * 60)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading model...")

    model = joblib.load(
        MODEL_PATH
    )

    print(
        f"Model loaded from: {MODEL_PATH}"
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print(
        "\nLoading sales data..."
    )

    sales = load_sales()

    print(
        f"Sales shape: {sales.shape}"
    )

    # --------------------------------------------------------
    # Select products
    # --------------------------------------------------------

    selected_items = select_top_products(
        sales,
        N_TOP_PRODUCTS
    )

    print(
        f"\nSelected {len(selected_items)} products."
    )

    # --------------------------------------------------------
    # Reshape
    # --------------------------------------------------------

    print(
        "\nReshaping sales data..."
    )

    sales_long = reshape_sales(
        sales,
        selected_items
    )

    print(
        f"Long-format shape: "
        f"{sales_long.shape}"
    )

    # --------------------------------------------------------
    # Calendar
    # --------------------------------------------------------

    print(
        "\nMerging calendar..."
    )

    calendar = load_calendar()

    sales_long = merge_calendar(
        sales_long,
        calendar
    )

    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    history = sales_long[
        sales_long["date"] <= "2016-04-24"
    ][
        [
            "item_id",
            "date",
            "sales"
        ]
    ].copy()

    # --------------------------------------------------------
    # Future dates
    # --------------------------------------------------------

    future_dates = pd.date_range(
        start=FORECAST_START,
        end=FORECAST_END,
        freq="D"
    )

    print(
        "\nForecast period:"
    )

    print(
        f"{FORECAST_START} "
        f"to "
        f"{FORECAST_END}"
    )

    print(
        f"Forecast days: "
        f"{len(future_dates)}"
    )

    # --------------------------------------------------------
    # Forecast
    # --------------------------------------------------------

    print(
        "\nGenerating recursive forecast..."
    )

    predictions = recursive_forecast(
        model=model,
        history=history,
        future_dates=future_dates
    )

    # --------------------------------------------------------
    # Save forecast
    # --------------------------------------------------------

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    predictions.to_csv(
        FORECAST_PATH,
        index=False
    )

    print(
        "\nForecast generated successfully."
    )

    print(
        f"Rows: {len(predictions)}"
    )

    print(
        f"Total forecasted units: "
        f"{predictions['prediction'].sum():,.2f}"
    )

    print(
        f"\nForecast saved to:"
    )

    print(
        FORECAST_PATH
    )

    print(
        "\nDone!"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    generate_forecast()