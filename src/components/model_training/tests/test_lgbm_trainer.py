from typing import Dict, Optional, OrderedDict

import pandas as pd
import pytest
from house_pricing.model_training.lgbm_trainer import train_model
from lightgbm import LGBMRegressor
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import RobustScaler


@pytest.fixture
def toy_dataset() -> pd.DataFrame:
    df, y = load_diabetes(return_X_y=True, as_frame=True)
    df["target"] = y
    return df


def test_train_model(
    toy_dataset: pd.DataFrame,
) -> None:
    model, _, _ = train_model(
        df=toy_dataset,
        target_colname="target",
        model_hparams={},
        fitting_params={},
        robust_scaling=False,
        sampling_seed=42, 
        train_size=0.7,
        val_size=0.2,

    )
    assert isinstance(model, LGBMRegressor)
    assert model.__sklearn_is_fitted__()


@pytest.mark.parametrize(
    "robust_scaling, expected_scaler",
    [
        (True, RobustScaler),
        (False, type(None)),
    ],
)
def test_train_scaler(
    toy_dataset: pd.DataFrame,
    robust_scaling: bool,
    expected_scaler: Optional[RobustScaler],
) -> None:
    model, _, scaler = train_model(
        df=toy_dataset,
        target_colname="target",
        model_hparams={
            "boosting_type": "gbdt",
            "num_leaves": 256,
            "learning_rate": 0.1,
            "n_estimators": 5000,
            "verbose": -1,
        },
        fitting_params = {
            "eval_metric": ["mse"],
        },
        robust_scaling=robust_scaling,
        sampling_seed=42, 
        train_size=0.7,
        val_size=0.2,

    )
    assert isinstance(model, LGBMRegressor)
    assert isinstance(scaler, expected_scaler)


def test_train_metrics(
    toy_dataset: pd.DataFrame,
) -> None:
    _, eval_metrics, _ = train_model(
        df=toy_dataset,
        target_colname="target",
        model_hparams={
            "boosting_type": "gbdt",
            "num_leaves": 256,
            "learning_rate": 0.1,
            "n_estimators": 5000,
            "verbose": -1,
        },
        fitting_params = {
            "eval_metric": ["mse"],
        },
        robust_scaling=False,
        sampling_seed=42, 
        train_size=0.7,
        val_size=0.2,

    )
    expected_training_metrics_keys = ["train", "val"]
    regression_metrics_expected_keys = ["train", "val", "test"]
    expected_metrics = ["mae", "mse", "mape"]
    assert isinstance(eval_metrics, Dict)
    assert list(eval_metrics.keys()) == ["training_metrics", "regression_metrics"]
    assert isinstance(eval_metrics["training_metrics"], Dict)
    assert isinstance(eval_metrics["regression_metrics"], Dict)
    assert list(eval_metrics["training_metrics"].keys()) == expected_training_metrics_keys
    assert list(eval_metrics["regression_metrics"].keys()) == regression_metrics_expected_keys
    for k in expected_training_metrics_keys:
        isinstance(eval_metrics["training_metrics"][k], OrderedDict)
    for k in regression_metrics_expected_keys:
        assert list(eval_metrics["regression_metrics"][k].keys()) == expected_metrics
        assert all([isinstance(v, float) for v in eval_metrics["regression_metrics"][k].values()])
        assert eval_metrics["regression_metrics"][k]["mape"] >= 0 and eval_metrics["regression_metrics"][k]["mape"] <= 1
