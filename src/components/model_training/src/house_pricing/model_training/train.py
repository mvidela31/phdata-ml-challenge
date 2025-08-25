import argparse
import inspect
import json
import os

import joblib
import pandas as pd
from house_pricing.model_training import train_model
from house_pricing.model_training.utils import parse_str_dict


def main() -> None:
    """
    Command line invoked function that trains the regression model.

    Trains a LightGBM regression model.
    """
    parser = argparse.ArgumentParser(description="Train a LightGBM regression model.")
    parser.add_argument(
        "--data-path",
        type=str,
        default=os.path.join(os.environ["SM_CHANNELS"], "dataset", "dataset_df.parquet"),
        help="Parquet file path of input dataset.",
    )
    parser.add_argument(
        "--target-colname",
        type=str,
        required=True,
        help="Column name of the regression target variable.",
    )
    parser.add_argument(
        "--model-hparams",
        type=str,
        default="{}",
        help="Regression model hyperparameters.",
    )
    parser.add_argument(
        "--fitting-params",
        type=str,
        default="{}",
        help="Regression model fitting parameters.",
    )
    parser.add_argument(
        "--robust-scaling",
        type=str,
        default="False",
        help="Whether to add a robust scaling preprocessing step.",
    )
    parser.add_argument(
        "--sampling-seed",
        type=str,
        default="42",
        help="Data splitting sampling seed.",
    )
    parser.add_argument(
        "--train-size",
        type=str,
        default="0.7",
        help="Percentage size of train set.",
    )
    parser.add_argument(
        "--val-size",
        type=str,
        default="0.1",
        help="Percentage size of validation set.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ["SM_MODEL_DIR"],
        help="Directory to save the trained model artifacts.",
    )

    # Arguments parsing
    args, _ = parser.parse_known_args()
    args = {k.replace("-", "_"): v for k, v in args.__dict__.items()}
    args = parse_str_dict(args)

    # Automatic argparse extraction
    required_args = set(inspect.signature(train_model).parameters.keys())
    training_args = {k: v for k, v in args.items() if k in required_args}
    training_args["df"] = pd.read_parquet(args["data_path"], engine="fastparquet")

    # Model training
    model, eval_metrics, scaler = train_model(**training_args)

    # Model artifacts saving
    if args["model_dir"] is not None:
        os.makedirs(args["model_dir"], exist_ok=True)
        joblib.dump(model, os.path.join(args["model_dir"], "model.joblib"))
        with open(os.path.join(args["model_dir"], "eval_metrics.json"), "w") as f:
            json.dump(eval_metrics, f)
        if scaler is not None:
            joblib.dump(model, os.path.join(args["model_dir"], "scaler.joblib"))


if __name__ == "__main__":
    main()
