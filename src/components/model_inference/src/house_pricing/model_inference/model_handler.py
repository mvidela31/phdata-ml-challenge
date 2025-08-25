import os
import shutil
import urllib.request
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from lightgbm import LGBMRegressor


class ModelHandler(object):
    """
    LightGBM regression model handler for house pricing.
    """

    def __init__(self, models_dir: str) -> None:
        """
        LightGBM model handler for house pricing.

        Parameters
        ----------
        models_dir: str
            Directory to store the models.
        """
        if os.path.exists(models_dir):
            shutil.rmtree(models_dir)
        os.makedirs(models_dir)
        self.models_dir: str = models_dir
        self.initialized: bool = False
        self.model: Optional[LGBMRegressor] = None
        self.version: int = 0
        self.latest_version: int = 0

    def check_model(self, model: Any) -> None:
        """
        Check if input model is a fitted LGBMRegressor.

        Parameters
        ----------
        model: Any
            A model to check.
        """
        assert isinstance(model, LGBMRegressor)
        assert model.__sklearn_is_fitted__()
        if self.initialized:
            assert model.feature_name_ == self.model.feature_name_

    def load_model(self, model_path: str) -> None:
        """
        Model loader.

        Parameters
        ----------
        model_path: str
            Model artifacts path.
        """
        new_model = joblib.load(model_path)
        self.check_model(new_model)
        self.model = new_model
        self.model_name = os.path.basename(model_path)
        self.latest_version += 1
        self.version = self.latest_version
        self.initialized = True

    def download_model(self, model_url: str) -> None:
        """
        Download a model from url.

        Parameters
        ----------
        model_url: str
            Model url to download.
        """
        filepath, _ = urllib.request.urlretrieve(
            url=model_url,
            filename=os.path.join(self.models_dir, f"model_v{self.latest_version + 1}.joblib"),
        )
        self.load_model(filepath)

    def update_model(self, model_version: int) -> None:
        """
        Swap the current model from another available model.

        Parameter
        ---------
        model_version: int
            Model version to load.
        """
        model_path = os.path.join(self.models_dir, f"model_v{model_version}.joblib")
        assert os.path.exists(model_path), "Model not found."
        self.model = joblib.load(model_path)
        self.version = model_version

    def delete_model(self, model_version: int) -> None:
        """
        Delete model an available model.

        Parameter
        ---------
        model_version: int
            Model version to delete.
        """
        assert model_version != self.version, "Cannot delete the model version on use."
        model_path = os.path.join(self.models_dir, f"model_v{model_version}.joblib")
        os.remove(model_path)

    def get_available_models(self) -> List[int]:
        """
        Return the available model versions.

        Returns
        -------
        List[int]
            Available model versions.
        """
        assert self.initialized, "No models available."
        model_versions = [
            int(os.path.basename(model_path)[7:-7])
            for model_path in os.listdir(self.models_dir)
        ]
        return model_versions
    
    def preprocess(
        self,
        data: Dict[str, Any],
        complementary_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Transform raw input into model input data.

        Parameters
        ----------
        data: Dict[str, Any]
            Request data.
        complementary_df: pd.DataFrame
            Complementary data.

        Returns
        -------
        pd.DataFrame
            Preprocessed input dataframe.
        """
        df = pd.DataFrame.from_records(data)
        df = df.merge(complementary_df, how="left", on="zipcode")
        df = df[self.model.feature_name_]
        return df

    def inference(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Internal model inference method.

        Parameters
        ----------
        df: pd.DataFrame
            Classification model input data.

        Returns
        -------
        pd.DataFrame
            Output dataframe with model predictions and model version.
        """
        preds_df = pd.DataFrame()
        preds_df["prediction"] = self.model.predict(df)
        return preds_df

    def postprocess(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Return inference dataframe as a dictionary.

        Parameters
        ----------
        df: pd.DataFrame
            Dataframe with model predictions and model version.

        Returns
        -------
        Dict[str, Any]
            Output dictionary with model predictions and model version.
        """
        predictions = df.to_dict(orient="records")
        output = {
            "predictions": predictions,
            "model_version": str(self.version),
        }
        return output

    def handle(
        self,
        data: Dict[str, Any],
        complementary_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Call preprocess, inference and post-process functions to perform the inference.

        Parameters
        ----------
        data: Dict[str, Any]
            Request data.
        complementary_df: pd.DataFrame
            Complementary data.

        Returns
        -------
        Dict[str, Any]
            Output dictionary with model predictions and model version.
        """
        assert self.initialized, "Model not initialized."
        model_input = self.preprocess(data, complementary_df)
        model_out = self.inference(model_input)
        output = self.postprocess(model_out)
        return output
