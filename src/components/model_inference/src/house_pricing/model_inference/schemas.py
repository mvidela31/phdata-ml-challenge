from typing import List

from pydantic import BaseModel


class InputPartialInstance(BaseModel):
    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    sqft_above: int
    sqft_basement: int
    zipcode: int

class InputPartialData(BaseModel):
    instances: List[InputPartialInstance]

class InputFullInstance(BaseModel):
    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    waterfront: int
    view: int
    condition: int
    grade: int
    sqft_above: int
    sqft_basement: int
    yr_built: int
    yr_renovated: int
    zipcode: int
    lat: float
    long: float
    sqft_living15: int
    sqft_lot15: int

class InputFullData(BaseModel):
    instances: List[InputFullInstance]

class InputModelDownload(BaseModel):
    model_url: str
    model_package: str

class InputModelUpdate(BaseModel):
    model_version: str
    model_package: str

class InputModelDelete(BaseModel):
    model_version: str
    model_package: str

class OutputInstance(BaseModel):
    prediction: float

class OutputData(BaseModel):
    predictions: List[OutputInstance]
    model_version: str
    model_package: str
