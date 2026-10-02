import pandas as pd

from src.features import (
    FEATURE_COLUMNS,
    create_demand_features,
    prepare_model_data,
)


def make_sample_data():

    dates = pd.date_range(
        start="2020-01-01",
        periods=40,
        freq="D"
    )

    data = pd.DataFrame(
        {
            "item_id": ["ITEM_1"] * 40,
            "date": dates,
            "sales": range(1, 41),
        }
    )

    return data


def test_feature_columns_exist():

    data = make_sample_data()

    result = create_demand_features(
        data
    )

    for column in FEATURE_COLUMNS:
        assert column in result.columns


def test_lag_features_are_created():

    data = make_sample_data()

    result = create_demand_features(
        data
    )

    assert result.loc[7, "lag_7"] == 1
    assert result.loc[28, "lag_28"] == 1


def test_feature_row_count():

    data = make_sample_data()

    result = prepare_model_data(
        data
    )

    assert len(result) == 12


def test_no_missing_model_features():

    data = make_sample_data()

    result = prepare_model_data(
        data
    )

    assert not result[
        FEATURE_COLUMNS
    ].isna().any().any()