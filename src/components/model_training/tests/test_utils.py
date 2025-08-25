from typing import Tuple

import pandas as pd
import pytest
from house_pricing.model_training.utils import parse_str_dict, train_val_test_split
from sklearn.datasets import load_diabetes


@pytest.fixture
def toy_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    return (X, y)


def test_parse_str_dict() -> None:
    src_dict = {
        "int": '1',
        "float": '0.1',
        "str": 'my_string',
        "bool": 'True',
        "null": 'null',
        "list": '[1, 0.1, "my_str", True, null, None, [1, 0.1]]',
        "dict": '{"a": 1, "b": 0.1, "c": "my_string", "d": True, "e": null, "f": [1, 0.1, True, null, [1, 0.1]]}',
    }
    parsed_dict = parse_str_dict(src_dict)
    expected_dict = {
        "int": 1,
        "float": 0.1,
        "str": "my_string",
        "bool": True,
        "null": None,
        "list": [1, 0.1, "my_str", True, None, "None", [1, 0.1]],
        "dict": {
            "a": 1,
            "b": 0.1,
            "c": "my_string",
            "d": True,
            "e": None,
            "f": [1, 0.1, True, None, [1, 0.1]],
        },
    }
    assert parsed_dict == expected_dict


@pytest.mark.parametrize(
    "train_size, val_size",
    [
        (0.5, 0.3),
        (0.8, 0.1),
        pytest.param(0.8, 0.8, marks=pytest.mark.xfail),
    ]
)
def test_train_val_test_split(
    toy_dataset: Tuple[pd.DataFrame, pd.Series],
    train_size: float,
    val_size: float,
) -> None:
    X, y = toy_dataset
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X=X,
        y=y,
        train_size=train_size,
        val_size=val_size,
    )
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_val, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_val, pd.Series)
    assert isinstance(y_test, pd.Series)
    assert all(X_train.isin(X))
    assert all(X_val.isin(X))
    assert all(X_test.isin(X))
    assert all(y_train.isin(y))
    assert all(y_val.isin(y))
    assert all(y_test.isin(y))
    assert len(X_train) + len(X_val) + len(X_test) == len(X)
    assert len(y_train) + len(y_val) + len(y_test) == len(y)
