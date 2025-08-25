from typing import Tuple, Union

import pandas as pd
import pytest
from house_pricing.model_training.eval import compute_regression_metrics
from house_pricing.model_training.utils import train_val_test_split
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator
from sklearn.datasets import load_diabetes
from sklearn.neighbors import KNeighborsRegressor


@pytest.fixture
def toy_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X=X,
        y=y,
        train_size=0.8,
        val_size=0.1,
        random_state=42,
    )
    return (X_train, X_val, X_test, y_train, y_val, y_test)


@pytest.mark.parametrize(
    "model_fn",
    [
        LGBMRegressor,
        KNeighborsRegressor,
    ]
)
def test_compute_regression_metrics(
    toy_dataset: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series],
    model_fn: Union[LGBMRegressor, BaseEstimator],
) -> None:
    X_train, X_val, X_test, y_train, y_val, y_test = toy_dataset
    model = model_fn().fit(X_train, y_train)
    regression_metrics = compute_regression_metrics(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        model=model,
    )
    expected_keys = ["train", "val", "test"]
    expected_metrics = ["mae", "mse", "mape"]
    assert list(regression_metrics.keys()) == expected_keys
    for k in expected_keys:
        assert list(regression_metrics[k].keys()) == expected_metrics
        assert all([isinstance(v, float) for v in regression_metrics[k].values()])
        assert regression_metrics[k]["mape"] >= 0 and regression_metrics[k]["mape"] <= 1
