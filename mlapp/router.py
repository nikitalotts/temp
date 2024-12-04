from fastapi import APIRouter, HTTPException, Body
from starlette import status
from starlette.responses import Response, JSONResponse

from mlapp.exceptions import ModelNotFoundError, Error
from mlapp.requests import *
from mlapp.model_manager import model_manager

router = APIRouter()


@router.post("/fit")
async def fit(request: FitRequest):
    x = request.dict
    return model_manager.fit(request.dict())
#
#

#
#
# @router.post("/unload")
# async def unload():
#     return {"message": "Inference space cleared"}
#
#


# @router.post("/predict")
# async def predict(request: PredictRequest):
#     return request.dict()

@router.post("/load")
async def load(request: LoadRequest):
    config = request.config
    model_manager.load(config.model_id, config.model_type, config.params)
    return Response(status_code=status.HTTP_204_NO_CONTENT)

@router.post("/unload")
async def unload():
    model_manager.unload()
    return Response(status_code=status.HTTP_204_NO_CONTENT)

@router.get("/get_status")
async def get_status():
    inference_models = model_manager.get_status()
    return JSONResponse(status_code=status.HTTP_200_OK, content={"inference_models": inference_models})


@router.get("/list_models")
async def list_models():
    models = model_manager.list_models()
    return JSONResponse(status_code=status.HTTP_200_OK, content={"loaded_models": models})


@router.post("/remove")
async def remove(request: RemoveModelRequest):
    model_manager.remove(request.model_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/remove_all")
async def remove_all():
    model_manager.remove_all()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
