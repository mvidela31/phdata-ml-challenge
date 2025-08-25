from typing import List

import pandas as pd
import pandas.api.types as ptypes
import pytest
from house_pricing.data_ingestion.data_ingestor import load_house_pricing_dataset


@pytest.mark.parametrize(
    "sales_column_selection",
    [
        [
            "price", 
            "bedrooms",
            "bathrooms",
            "sqft_living",
            "sqft_lot",
            "floors",
            "sqft_above",
            "sqft_basement",
            "zipcode",
        ],
        [
            "price",
            "bedrooms",
            "bathrooms",
            "sqft_living",
            "sqft_lot",
            "floors",
            "waterfront",
            "view",
            "condition",
            "grade",
            "sqft_above",
            "sqft_basement",
            "yr_built",
            "yr_renovated",
            "zipcode",
            "lat",
            "long",
            "sqft_living15",
            "sqft_lot15",
        ],
        pytest.param(
            [
                "price", 
                "bedrooms",
                "bathrooms",
                "sqft_living",
                "sqft_lot",
                "floors",
                "sqft_above",
                "sqft_basement",
            ],
            marks=pytest.mark.xfail,
        ),
        pytest.param(
            [
                "non_existent_parameter", 
                "bedrooms",
                "bathrooms",
                "sqft_living",
                "sqft_lot",
                "floors",
                "sqft_above",
                "sqft_basement",
                "zipcode",
            ],
            marks=pytest.mark.xfail,
        ),
    ]
)
def test_load_house_pricing_dataset_output(sales_column_selection: List[str]) -> None:
    df = load_house_pricing_dataset(
        sales_column_selection=sales_column_selection
    )
    expected_cols = sales_column_selection.copy()
    expected_cols.remove("zipcode")
    assert isinstance(df, pd.DataFrame)
    assert "zipcode" not in df.columns
    assert set(expected_cols).issubset(df.columns.to_list())
    assert all(ptypes.is_numeric_dtype(df[col]) for col in df.columns)
