from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from yaml import Loader, load


def parse_str_dict(
    dict: Dict[str, str],
) -> Dict[str, Any]:
    """
    Parses a dictionary with string values by evaluating its \
    Python expression using the yaml loader.

    Parameters
    ----------
    dict: Dict[str, str]
        Dictionary with string values.

    Returns
    -------
    Dict[str, Any]
        Dictionary with evaluated values.
    """
    for k, v in dict.items():
        dict[k] = load(v, Loader=Loader)
    return dict


def train_val_test_split(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.DataFrame],
    train_size: float,
    val_size: float,
    random_state: int = 42,
    shuffle: bool = True,
) -> Tuple[
    Union[np.ndarray, pd.DataFrame],
    Union[np.ndarray, pd.DataFrame],
    Union[np.ndarray, pd.DataFrame],
    Union[np.ndarray, pd.DataFrame],
    Union[np.ndarray, pd.DataFrame],
    Union[np.ndarray, pd.DataFrame],
]:
    """
    Splits the input array into three partitions: Train, validation, and test.

    Parameters
    ----------
    X: Union[np.ndarray, pd.DataFrame]
        Input dataset.
    y: Union[np.ndarray, pd.DataFrame]
        Input labels.
    train_size: float
        Train dataset split size proportion. The test dataset size \
        proportion is calculated as: `test_size = 1 - (train_size + val_size)`.
    val_size: float
        Validation dataset split size proportion. The test dataset size \
        proportion is calculated as: `test_size = 1 - (train_size + val_size)`.
    random_state: int
        Dataset splitting rng seed.
    shuffle: bool
        Whether or not to shuffle the data before splitting.

    Returns
    -------
    Tuple[
        Union[np.ndarray, pd.DataFrame],
        Union[np.ndarray, pd.DataFrame],
        Union[np.ndarray, pd.DataFrame],
        Union[np.ndarray, pd.DataFrame],
        Union[np.ndarray, pd.DataFrame],
        Union[np.ndarray, pd.DataFrame],
    ]
        Train, validation, and test datasets and labels splits.
    """
    assert train_size + val_size < 1.0, "The sum of train and validation sizes is higher than 1.0."
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        train_size=train_size,
        random_state=random_state,
        shuffle=shuffle,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_val,
        y_val,
        train_size=val_size / (1.0 - train_size),
        random_state=random_state,
        shuffle=shuffle,
    )
    return (X_train, X_val, X_test, y_train, y_val, y_test)
