from typing import Dict, Union

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)


def compute_regression_metrics(
    X_train: Union[np.ndarray, pd.DataFrame],
    X_val: Union[np.ndarray, pd.DataFrame],
    X_test: Union[np.ndarray, pd.DataFrame],
    y_train: Union[np.ndarray, pd.DataFrame],
    y_val: Union[np.ndarray, pd.DataFrame],
    y_test: Union[np.ndarray, pd.DataFrame],
    model: Union[LGBMRegressor, BaseEstimator],
) -> Dict[str, Dict[str, float]]:
    """
    Computes several regression metrics for train, validation, and test datasets.

    Parameters
    ----------
    X_train: Union[np.ndarray, pd.DataFrame]
        Train dataset.
    X_val: Union[np.ndarray, pd.DataFrame]
        Validation dataset.
    X_test: Union[np.ndarray, pd.DataFrame]
        Test dataset.
    y_train: Union[np.ndarray, pd.DataFrame]
        Train labels.
    y_val: Union[np.ndarray, pd.DataFrame]
        Validation labels.
    y_test: Union[np.ndarray, pd.DataFrame]
        Test labels.
    model: Union[lightgbm.LGBMRegressor, sklearn.base.Estimator]
        Regression model.

    Returns
    -------
    Dict[str, List[List[float]]]
        Train, validation, and test classification metrics.
    """
    settings = [
        {"split": "train", "y_true": y_train, "y_pred": model.predict(X_train)},
        {"split": "val", "y_true": y_val, "y_pred": model.predict(X_val)},
        {"split": "test", "y_true": y_test, "y_pred": model.predict(X_test)},
    ]
    regression_metrics = {}
    for setting in settings:
        regression_metrics[setting["split"]] = {
            "mae": mean_absolute_error(
                y_true=setting["y_true"], y_pred=setting["y_pred"]
            ),
            "mse": mean_squared_error(
                y_true=setting["y_true"], y_pred=setting["y_pred"],
            ),
            "mape": mean_absolute_percentage_error(
                y_true=setting["y_true"], y_pred=setting["y_pred"],
            ),
        }
    return regression_metrics
