from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.features import FEATURE_COLUMNS, prepare_model_data


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

SALES_PATH = DATA_DIR / "sales_train_validation.csv"
MODEL_PATH = MODEL_DIR / "demand_forecaster.joblib"


# ============================================================
# CONFIGURATION
# ============================================================

STORE_ID = "CA_1"

TRAIN_END_DATE = "2016-04-24"

N_TOP_PRODUCTS = 20

RANDOM_STATE = 42


# ============================================================
# LOAD SALES DATA
# ============================================================

def load_sales_data():

    sales = pd.read_csv(
        SALES_PATH
    )

    return sales


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
# RESHAPE SALES DATA
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
# LOAD CALENDAR
# ============================================================

def load_calendar():

    calendar = pd.read_csv(
        DATA_DIR / "calendar.csv"
    )

    calendar["date"] = pd.to_datetime(
        calendar["date"]
    )

    return calendar


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
# TRAIN MODEL
# ============================================================

def train_model():

    print("=" * 60)
    print("RETAIL DEMAND FORECASTER")
    print("=" * 60)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("\nLoading sales data...")

    sales = load_sales_data()

    print(
        f"Sales shape: {sales.shape}"
    )

    # --------------------------------------------------------
    # Select products
    # --------------------------------------------------------

    print(
        f"\nSelecting top {N_TOP_PRODUCTS} "
        f"products from {STORE_ID}..."
    )

    selected_items = select_top_products(
        sales,
        N_TOP_PRODUCTS
    )

    print(
        "Selected products:"
    )

    for item in selected_items:
        print(
            f"  - {item}"
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
    # Features
    # --------------------------------------------------------

    print(
        "\nCreating forecasting features..."
    )

    model_data = prepare_model_data(
        sales_long
    )

    # --------------------------------------------------------
    # Training data
    # --------------------------------------------------------

    train_data = model_data[
        model_data["date"] <= TRAIN_END_DATE
    ].copy()

    X_train = train_data[
        FEATURE_COLUMNS
    ]

    y_train = train_data[
        "sales"
    ]

    print(
        f"\nTraining rows: "
        f"{len(train_data):,}"
    )

    print(
        f"Feature matrix: "
        f"{X_train.shape}"
    )

    print(
        f"Training period: "
        f"{train_data['date'].min().date()} "
        f"to "
        f"{train_data['date'].max().date()}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print(
        "\nTraining HistGradientBoostingRegressor..."
    )

    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=RANDOM_STATE
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "Model training complete."
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    print(
        f"\nModel saved to:"
    )

    print(
        MODEL_PATH
    )

    print(
        "\nDone!"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train_model()