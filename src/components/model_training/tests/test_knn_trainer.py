from typing import Dict, Union

import pandas as pd
import pytest
from house_pricing.model_training.knn_trainer import train_model
from sklearn.datasets import load_diabetes
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted


@pytest.fixture
def toy_dataset() -> pd.DataFrame:
    df, y = load_diabetes(return_X_y=True, as_frame=True)
    df["target"] = y
    return df


@pytest.mark.parametrize(
    "robust_scaling, expected_model",
    [
        (False, KNeighborsRegressor),
        (True, Pipeline),
    ]
)
def test_train_model(
    toy_dataset: pd.DataFrame,
    robust_scaling: bool,
    expected_model: Union[KNeighborsRegressor, Pipeline],
) -> None:
    model, _ = train_model(
        df=toy_dataset,
        target_colname="target",
        robust_scaling=robust_scaling,
        sampling_seed=42, 
        train_size=0.7,
        val_size=0.2,

    )
    assert isinstance(model, expected_model)
    check_is_fitted(model)


def test_train_metrics(
    toy_dataset: pd.DataFrame,
) -> None:
    _, eval_metrics = train_model(
        df=toy_dataset,
        target_colname="target",
        robust_scaling=True,
        sampling_seed=42, 
        train_size=0.7,
        val_size=0.2,

    )
    regression_metrics_expected_keys = ["train", "val", "test"]
    expected_metrics = ["mae", "mse", "mape"]
    assert isinstance(eval_metrics, Dict)
    assert list(eval_metrics.keys()) == ["regression_metrics"]
    assert isinstance(eval_metrics["regression_metrics"], Dict)
    assert list(eval_metrics["regression_metrics"].keys()) == regression_metrics_expected_keys
    for k in regression_metrics_expected_keys:
        assert list(eval_metrics["regression_metrics"][k].keys()) == expected_metrics
        assert all([isinstance(v, float) for v in eval_metrics["regression_metrics"][k].values()])
        assert eval_metrics["regression_metrics"][k]["mape"] >= 0 and eval_metrics["regression_metrics"][k]["mape"] <= 1
