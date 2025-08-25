import os
from typing import Dict, List
from urllib.parse import unquote

import pandas as pd
from fastapi import FastAPI, HTTPException
from house_pricing.model_inference.model_handler import ModelHandler
from house_pricing.model_inference.schemas import (
    InputFullData,
    InputModelDelete,
    InputModelDownload,
    InputModelUpdate,
    InputPartialData,
    OutputData,
)

app = FastAPI()
demographics_df = pd.read_csv(os.path.join(
    os.path.dirname(__file__), "data", "zipcode_demographics.csv",
))
MODEL_PACKAGES: Dict[str, ModelHandler] = {
    "partial": ModelHandler(
        models_dir=os.path.join(os.path.dirname(__file__), "models_partial")
    ),
    "full": ModelHandler(
        models_dir=os.path.join(os.path.dirname(__file__), "models_full")
    ),
}

@app.get("/ping", status_code=200)
async def ping() -> Dict[str, str]:
    """
    Returns the application health status.

    Returns
    -------
    Dict[str, str]
        Application healthy status.
    """
    return {"status": "ok"}


@app.get("/versions/{model_package}", status_code=200)
async def get_versions(model_package: str) -> Dict[str, List[int]]:
    """
    Returns the available model versions

    Parameters
    ----------
    model_package: str
        Model package.

    Returns
    -------
    Dict[str, str]
        Available model versions
    """
    try:
        model_versions = MODEL_PACKAGES[model_package].get_available_models()
        return {"model_versions": model_versions}
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err)) from err


@app.put("/download_model")
async def download_model(data: InputModelDownload) -> Dict[str, str]:
    """
    Download a model from URL.

    Parameters
    ----------
    model_url: str
        Model url.
    model_package: str
        Model package.

    Returns
    -------
    Dict[str, str]
        Model download success status.
    """
    data_dump = data.model_dump()
    try:
        model_url = unquote(unquote(data_dump["model_url"]))
        model_package = data_dump["model_package"]
        MODEL_PACKAGES[model_package].download_model(model_url)
        return {"status": "Model downloaded successfully"}
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err)) from err


@app.put("/update_model")
async def update_model(data: InputModelUpdate) -> Dict[str, str]:
    """
    Update the regression model version.

    Parameters
    ----------
    model_version: str
        Model version.
    model_package: str
        Model package.

    Returns
    -------
    Dict[str, str]
        Model update success status.
    """
    data_dump = data.model_dump()
    try:
        model_version = data_dump["model_version"]
        model_package = data_dump["model_package"]
        MODEL_PACKAGES[model_package].update_model(int(model_version))
        return {"status": "Model updated successfully"}
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err)) from err
    

@app.delete("/delete_model")
async def delete_model(data: InputModelDelete) -> Dict[str, str]:
    """
    Delete the regression model version.

    Parameters
    ----------
    model_version: str
        Model version.
    model_package: str
        Model package.

    Returns
    -------
    Dict[str, str]
        Model delete success status.
    """
    data_dump = data.model_dump()
    try:
        model_version = data_dump["model_version"]
        model_package = data_dump["model_package"]
        MODEL_PACKAGES[model_package].delete_model(int(model_version))
        return {"status": "Model deleted successfully"}
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err)) from err


@app.post("/invocations_partial", response_model=OutputData)
async def invocations_partial(data: InputPartialData) -> OutputData:
    """
    Returns the predictions for the incoming request 
    using the "partial" regression model package.

    Parameters
    ----------
    data: InputPartialData
        Input data instances.
    
    Returns
    -------
    OutputData
        Output predictions with model metadata.
    """
    data_dump = data.model_dump()
    try:
        instances = data_dump["instances"]
        output = MODEL_PACKAGES["partial"].handle(instances, demographics_df)
        output.update({"model_package": "partial"})
        return output
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err)) from err


@app.post("/invocations_full", response_model=OutputData)
async def invocations_full(data: InputFullData) -> OutputData:
    """
    Returns the predictions for the incoming request 
    using the "full" regression model package.

    Parameters
    ----------
    data: InputFullData
        Input data instances.

    Returns
    -------
    OutputData
        Output predictions with model metadata.
    """
    data_dump = data.model_dump()
    try:
        instances = data_dump["instances"]
        output = MODEL_PACKAGES["full"].handle(instances, demographics_df)
        output.update({"model_package": "full"})
        return output
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err)) from err
