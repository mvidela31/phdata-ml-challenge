# NOTE: Only for replication purpouses. It should be deleted.
import logging
from pprint import pformat
from typing import Any, Dict, Tuple, Union

import pandas as pd
from house_pricing.model_training.regressor_eval import compute_regression_metrics
from house_pricing.model_training.utils import train_val_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import RobustScaler

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def train_model(
    df: pd.DataFrame,
    target_colname: str,
    robust_scaling: bool = True,
    sampling_seed: int = 36, 
    train_size: float = 0.6,
    val_size: float = 0.2,
) -> Tuple[Union[KNeighborsRegressor, Pipeline], Dict[str, Any]]:
    """
    Trains a KNeighbors regression model.

    Parameters
    ----------
    df: pd.DataFrame
        Input dataset.
    target_colname: str
        Column name of the classification target variable.
    robust_scaling: bool
        Whether to add a robust scaling preprocessing step.
    sampling_seed: int
        A randomization seed used when sampling the subset of healthy clients.
        For replication purposes.
    train_size: float
        Relative size of training data to produce.
    val_size: float
        Relative size of validation data to produce.

    Returns
    -------
    Tuple[Union[KNeighborsRegressor, Pipeline], Dict[str, Dict[str, Any]]
        Trained KNeighbors regression model with its evaluation metrics.
    """
    # Data preparation
    df = df.copy()
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X=df.drop(columns=[target_colname]),
        y=df[[target_colname]],
        train_size=train_size,
        val_size=val_size,
        random_state=sampling_seed,
        shuffle=True,
    )
    
    # Model training
    model = KNeighborsRegressor()
    if robust_scaling:
        model = make_pipeline(RobustScaler(), model)
    model = model.fit(X_train, y_train)

    # Evaluation metrics
    eval_metrics = {
        "regression_metrics": compute_regression_metrics(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            model=model,
        ),
    }
    logger.info(f"Regression Metrics:\n{pformat(eval_metrics['regression_metrics'])}")
    return (model, eval_metrics)
