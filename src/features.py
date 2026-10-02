import pandas as pd


FEATURE_COLUMNS = [
    "lag_1",
    "lag_7",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_std_28",
    "day_of_week",
    "month",
    "week_of_year",
    "is_weekend",
]


def create_demand_features(
    sales_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Create lag, rolling-window, and calendar features
    for demand forecasting.

    Required columns:
        item_id
        date
        sales

    Returns:
        DataFrame containing the original data plus
        forecasting features.
    """

    data = sales_data.copy()

    data["date"] = pd.to_datetime(
        data["date"]
    )

    data = data.sort_values(
        ["item_id", "date"]
    ).reset_index(drop=True)

    grouped_sales = (
        data.groupby("item_id")["sales"]
    )

    # Lag features
    data["lag_1"] = (
        grouped_sales.shift(1)
    )

    data["lag_7"] = (
        grouped_sales.shift(7)
    )

    data["lag_28"] = (
        grouped_sales.shift(28)
    )

    # Rolling features
    shifted_sales = (
        grouped_sales.shift(1)
    )

    data["rolling_mean_7"] = (
        shifted_sales
        .rolling(7)
        .mean()
        .reset_index(
            level=0,
            drop=True
        )
    )

    data["rolling_mean_28"] = (
        shifted_sales
        .rolling(28)
        .mean()
        .reset_index(
            level=0,
            drop=True
        )
    )

    data["rolling_std_28"] = (
        shifted_sales
        .rolling(28)
        .std()
        .reset_index(
            level=0,
            drop=True
        )
    )

    # Calendar features
    data["day_of_week"] = (
        data["date"].dt.dayofweek
    )

    data["month"] = (
        data["date"].dt.month
    )

    data["week_of_year"] = (
        data["date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    data["is_weekend"] = (
        data["day_of_week"] >= 5
    ).astype(int)

    return data


def prepare_model_data(
    sales_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Create forecasting features and remove rows
    that do not have enough historical observations.
    """

    data = create_demand_features(
        sales_data
    )

    data = data.dropna(
        subset=FEATURE_COLUMNS
    ).copy()

    return data