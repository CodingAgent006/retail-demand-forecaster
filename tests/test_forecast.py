import pandas as pd

from src.forecast import recursive_forecast


def make_history():
    dates = pd.date_range(
        start="2020-01-01",
        periods=40,
        freq="D"
    )

    rows = []

    for item_id in ["ITEM_1", "ITEM_2"]:
        for date, sales in zip(dates, range(1, 41)):
            rows.append(
                {
                    "item_id": item_id,
                    "date": date,
                    "sales": sales,
                }
            )

    return pd.DataFrame(rows)


class DummyModel:
    def predict(self, X):
        return [10.0] * len(X)


def test_forecast_row_count():
    history = make_history()

    future_dates = pd.date_range(
        start="2020-02-10",
        periods=28,
        freq="D"
    )

    predictions = recursive_forecast(
        model=DummyModel(),
        history=history,
        future_dates=future_dates
    )

    assert len(predictions) == 2 * 28


def test_forecast_contains_all_items():
    history = make_history()

    future_dates = pd.date_range(
        start="2020-02-10",
        periods=28,
        freq="D"
    )

    predictions = recursive_forecast(
        model=DummyModel(),
        history=history,
        future_dates=future_dates
    )

    assert set(predictions["item_id"]) == {
        "ITEM_1",
        "ITEM_2",
    }


def test_forecast_has_correct_dates():
    history = make_history()

    future_dates = pd.date_range(
        start="2020-02-10",
        periods=28,
        freq="D"
    )

    predictions = recursive_forecast(
        model=DummyModel(),
        history=history,
        future_dates=future_dates
    )

    assert predictions["date"].min() == future_dates.min()
    assert predictions["date"].max() == future_dates.max()


def test_forecast_is_non_negative():
    history = make_history()

    future_dates = pd.date_range(
        start="2020-02-10",
        periods=28,
        freq="D"
    )

    predictions = recursive_forecast(
        model=DummyModel(),
        history=history,
        future_dates=future_dates
    )

    assert (predictions["prediction"] >= 0).all()