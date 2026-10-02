from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.forecast import recursive_forecast
from src.features import prepare_model_data
from src.train import (
    STORE_ID,
    N_TOP_PRODUCTS,
    load_calendar,
    load_sales_data,
    merge_calendar,
    reshape_sales,
    select_top_products,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "demand_forecaster.joblib"
REPORTS_DIR = PROJECT_ROOT / "reports"
RESULTS_PATH = REPORTS_DIR / "model_evaluation.csv"


VALIDATION_WINDOWS = [
    ("2016-02-01", "2016-02-28"),
    ("2016-02-29", "2016-03-27"),
    ("2016-03-28", "2016-04-24"),
]


def evaluate_window(model, sales_data, validation_start, validation_end):
    validation_start = pd.Timestamp(validation_start)
    validation_end = pd.Timestamp(validation_end)

    history_end = validation_start - pd.Timedelta(days=1)

    history = sales_data[
        sales_data["date"] <= history_end
    ][["item_id", "date", "sales"]].copy()

    future_dates = pd.date_range(
        validation_start,
        validation_end,
        freq="D"
    )

    predictions = recursive_forecast(
        model=model,
        history=history,
        future_dates=future_dates,
    )

    actuals = sales_data[
        sales_data["date"].between(
            validation_start,
            validation_end
        )
    ][["item_id", "date", "sales"]].copy()

    comparison = predictions.merge(
        actuals,
        on=["item_id", "date"],
        how="left",
    )

    mae = mean_absolute_error(
        comparison["sales"],
        comparison["prediction"],
    )

    rmse = np.sqrt(
        mean_squared_error(
            comparison["sales"],
            comparison["prediction"],
        )
    )

    return {
        "validation_start": validation_start.date(),
        "validation_end": validation_end.date(),
        "mae": mae,
        "rmse": rmse,
        "actual_total": comparison["sales"].sum(),
        "predicted_total": comparison["prediction"].sum(),
    }


def main():
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    print("\nLoading model...")
    model = joblib.load(MODEL_PATH)

    print("Loading sales data...")
    sales = load_sales_data()

    print(f"Sales shape: {sales.shape}")

    print(f"\nSelecting top {N_TOP_PRODUCTS} products from {STORE_ID}...")
    selected_items = select_top_products(
        sales,
        N_TOP_PRODUCTS
    )

    print("\nReshaping sales...")
    sales_long = reshape_sales(
        sales,
        selected_items
    )

    print(f"Long-format shape: {sales_long.shape}")

    print("\nMerging calendar...")
    calendar = load_calendar()

    sales_long = merge_calendar(
        sales_long,
        calendar
    )

    print("\nPreparing evaluation data...")
    sales_data = prepare_model_data(
        sales_long
    )

    results = []

    print("\nRunning validation windows...")

    for validation_start, validation_end in VALIDATION_WINDOWS:

        print(
            f"\nEvaluating "
            f"{validation_start} → {validation_end}"
        )

        result = evaluate_window(
            model=model,
            sales_data=sales_data,
            validation_start=validation_start,
            validation_end=validation_end,
        )

        results.append(result)

        print(f"  MAE: {result['mae']:.4f}")
        print(f"  RMSE: {result['rmse']:.4f}")
        print(f"  Actual units: {result['actual_total']:.0f}")
        print(f"  Predicted units: {result['predicted_total']:.2f}")

    results_df = pd.DataFrame(results)

    average_row = {
        "validation_start": "AVERAGE",
        "validation_end": "",
        "mae": results_df["mae"].mean(),
        "rmse": results_df["rmse"].mean(),
        "actual_total": results_df["actual_total"].mean(),
        "predicted_total": results_df["predicted_total"].mean(),
    }

    results_df = pd.concat(
        [
            results_df,
            pd.DataFrame([average_row]),
        ],
        ignore_index=True,
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False
    )

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    print("\nResults:")
    print(results_df.to_string(index=False))

    print(f"\nSaved to:")
    print(RESULTS_PATH)


if __name__ == "__main__":
    main()