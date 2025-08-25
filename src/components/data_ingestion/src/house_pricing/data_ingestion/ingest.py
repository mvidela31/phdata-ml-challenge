import argparse
import inspect
import os
from pathlib import Path

from house_pricing.data_ingestion import load_house_pricing_dataset
from house_pricing.data_ingestion.utils import parse_str_dict


def main() -> None:
    """
    Command line invoked function that Loads the house pricing dataset.
    """
    parser = argparse.ArgumentParser(description="Loads the house pricing dataset.")
    parser.add_argument(
        "--sales-column-selection",
        type=str,
        required=True,
        help="Columns of house sales dataset to select.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=os.path.join(os.environ["SM_PROCESS_DIR"], "dataset", "dataset_df.parquet"),
        help="Loaded dataset path.",
    )

    # Arguments parsing
    args, _ = parser.parse_known_args()
    args = {k.replace("-", "_"): v for k, v in args.__dict__.items()}
    args = parse_str_dict(args)

    # Automatic argparse extraction
    required_args = set(inspect.signature(load_house_pricing_dataset).parameters.keys())
    ingestion_args = {k: v for k, v in args.items() if k in required_args}

    # Data ingestion
    df = load_house_pricing_dataset(**ingestion_args)

    # Outputs saving
    Path(args["output_path"]).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args["output_path"], index=False, engine="fastparquet")
    

if __name__ == "__main__":
    main()
