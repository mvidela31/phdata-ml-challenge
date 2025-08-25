import logging
from pprint import pformat
from typing import Any, Dict, Optional, Tuple, Union

import lightgbm
import pandas as pd
from house_pricing.model_training.eval import compute_regression_metrics
from house_pricing.model_training.utils import train_val_test_split
from lightgbm import LGBMRegressor
from sklearn.preprocessing import RobustScaler

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# TODO: Add hyperparameters tuning.
def train_model(
    df: pd.DataFrame,
    target_colname: str,
    model_hparams: Dict[str, Union[str, int, float, bool]] = {},
    fitting_params: Dict[str, Union[str, int, float, bool]] = {},
    robust_scaling: bool = False,
    sampling_seed: int = 36, 
    train_size: float = 0.6,
    val_size: float = 0.2,
) -> Tuple[LGBMRegressor, Dict[str, Any], Optional[RobustScaler]]:
    """
    Trains a LightGBM regression model.

    Parameters
    ----------
    df: pd.DataFrame
        Input dataset.
    target_colname: str
        Column name of the classification target variable.
    model_hparams: Dict[str, Union[str, int, float, bool]]
        Classification model hyperparameters.
    fitting_params: Dict[str, Union[str, int, float, bool]]
        Classification model fitting parameters.
    robust_scaling: bool
        Whether to add a robust scaling preprocessing step.
    sampling_seed: int
        Data splitting sampling seed.
    train_size: float
        Relative size of training set.
    val_size: float
        Relative size of validation set.

    Returns
    -------
    Tuple[LGBMRegressor, Dict[str, Dict[str, Any]]
        Trained LightGBM regression model, the evaluation metrics, and optionally the scaler model.
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

    # Model initialization
    model = LGBMRegressor(**model_hparams)
    if robust_scaling:
        scaler = RobustScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
    else:
        scaler = None
    
    # Model training
    training_metrics = {}
    fitting_params.update({
        "X": X_train,
        "y": y_train.squeeze(),
        "eval_set": [(X_train, y_train.squeeze()), (X_val, y_val.squeeze())],
        "eval_names": ["train", "val"],
        "callbacks": [
            lightgbm.log_evaluation(10),
            lightgbm.record_evaluation(training_metrics),
            lightgbm.early_stopping(30),
        ],
    })
    model.fit(**fitting_params)

    # Evaluation metrics
    eval_metrics = {
        "training_metrics": training_metrics,
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
    return (model, eval_metrics, scaler)
