from typing import List

import pandas as pd
from house_pricing.data_ingestion.constants import (
    DEMOGRAPHIC_DATA_PATH,
    HOUSE_DATA_PATH,
)


# TODO: Load the dataset from a proper SQL database.
def load_house_pricing_dataset(
    sales_column_selection: List[str],
) -> pd.DataFrame:
    """
    Loads the house pricing dataset.

    Parameters
    ----------
    sales_column_selection: List[str]
        Columns of house sales dataset to select.

    Returns
    -------
    pd.DataFrame
        House pricing dataset.
    """
    houses_df = pd.read_csv(
        HOUSE_DATA_PATH,
        usecols=sales_column_selection,
        dtype={"zipcode": str},
    )
    demographics_df = pd.read_csv(
        DEMOGRAPHIC_DATA_PATH,
        dtype={"zipcode": str},
    )
    houses_df = houses_df.merge(
        right=demographics_df,
        how="left",
        on="zipcode",
    ).drop(columns="zipcode")
    return houses_df 
