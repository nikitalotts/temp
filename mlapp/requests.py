from mlapp.enums import ModelType
from pydantic import BaseModel
from typing import List, Dict, Optional


class ModelConfig(BaseModel):
    model_id: str
    model_type: ModelType
    params: Optional[Dict] = {}


class FitRequest(BaseModel):
    X: List[List[float]]
    y: List[float]
    config: ModelConfig


class LoadRequest(BaseModel):
    config: ModelConfig


class PredictRequest(BaseModel):
    X: List[List[float]]


class RemoveModelRequest(BaseModel):
    model_id: str
