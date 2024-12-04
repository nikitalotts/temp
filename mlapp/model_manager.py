import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Optional, List
import uuid
import multiprocessing
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.datasets import make_classification, make_regression

from mlapp.config import settings
from mlapp.enums import ModelType
from mlapp.exceptions import NoFreeCoresError, InvalidModelTypeError, ModelNotFoundError, ModelAlreadyExistsError, \
    MaxModelsReachedError, MismatchModelTypeError, ModelAlreadyLoadedError


class ModelConfig:
    def __init__(self, model_id: str, model_type: ModelType, params: Optional[Dict]):
        self.model_id = model_id
        self.model_type = model_type
        self.params = params
        self.path = None
        self.model = None


class ModelManager:
    def __init__(self, models_path, num_cores, max_inference_models):
        self.models_path = models_path
        self.num_cores = num_cores - 1
        self.max_inference_models = max_inference_models

        self.model_bank: Dict[ModelConfig] | Dict  = {}
        self.inference_models: Dict[ModelConfig] | Dict = {}
        self.lock = multiprocessing.Lock()
        self.executor = ProcessPoolExecutor(max_workers=self.num_cores)
        self.active_processes = 0

        # Генерация данных и обучение моделей
        self._initialize_models()

    def inc_processes_num(self, num=1):
        with self.lock:
            if self.active_processes + num > self.num_cores:
                raise NoFreeCoresError()
            self.active_processes += num

    def dec_processes_num(self, num=1):
        with self.lock:
            self.active_processes -= num

    def _initialize_models(self):
        X, y = make_classification(n_samples=100, n_features=20)

        self._train_model(
            str(uuid.uuid4()),
            ModelType.classification,
            {},
            X,
            y
        )
        print("ger1")
        X, y = make_regression(n_samples=100, n_features=20)
        
        self._train_model(
            str(uuid.uuid4()),
            ModelType.regression,
            {},
            X,
            y
        )
        print("ger2")
    def _train_model(
            self, 
            model_id: str,
            model_type: ModelType,
            params: Optional[Dict],
            X: List[List[float]],
            y: List[float]
    ):
        X = np.array(X) if isinstance(X, list) else X
        y = np.array(y) if isinstance(y, list) else y
        
        model = self._get_model_by_type(model_type, params)

        self.inc_processes_num()

        future = self.executor.submit(
            ModelManager._train_model_in_process,
            model, X, y, self.models_path, model_id)

        model_config = ModelConfig(model_id, model_type, params)
        future.add_done_callback(lambda fut: self._handle_future(fut, model_id, model_config))
        self.dec_processes_num()

    @staticmethod
    def _train_model_in_process(
            model,
            X: np.ndarray,
            y: np.ndarray,
            model_path: str,
            model_id: str
    ):
        model.fit(X, y)

        path = ModelManager.save_model(model, model_path, model_id)

        return path

    @staticmethod
    def save_model(model, model_path, model_id):
        path = os.path.join(model_path, model_id)
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        return path

    def _handle_future(self, future, model_id, model_config):
        model_path = future.result()
        model_config.path = model_path
        with self.lock:
            self.model_bank[model_id] = model_config
            print(f"Model {model_id} save to {model_path}")
        print(self.model_bank)

    def unload(self):
        for model_config in self.inference_models.values():
            model_config.model = None
            self.model_bank[model_config.model_id] = model_config
        del self.inference_models
        self.inference_models = {}

    def get_status(self):
        inference_models = []
        for model_id, model_config in self.inference_models.items():
            d = {
                'model_id': model_config.model_id,
                'model_type': model_config.model_type.value,
                'params': model_config.params
            }
            inference_models.append(d)

        return inference_models

    def list_models(self):
        loaded_models = []
        for model_id, model_config in self.model_bank.items():
            d = {
                'model_id': model_config.model_id,
                'model_type': model_config.model_type.value,
                'params': model_config.params
            }
            loaded_models.append(d)

        return loaded_models
        #return list(self.model_bank.keys())

    def remove(self, model_id):
        if model_id not in self.model_bank:
            raise ModelNotFoundError(model_id)

        model_path = self.model_bank[model_id].path
        del self.model_bank[model_id]
        if os.path.exists(model_path):
            if self.active_processes + 1 <= self.num_cores:
                _ = self.executor.submit(os.remove, model_path)
            else:
                os.remove(model_path)

    def remove_all(self):
        for model in list(self.model_bank.values()):
            self.remove(model.model_id)

    def load(self, model_id: str, model_type: ModelType, params: Dict):
        if model_id in self.inference_models:
            raise ModelAlreadyLoadedError(model_id)

        if model_id not in self.model_bank:
            raise ModelNotFoundError(model_id)

        if len(self.inference_models) + 1 > self.max_inference_models:
            raise MaxModelsReachedError()

        self._load_to_inference(model_id, model_type, params)

    def _get_model_by_type(self, model_type: ModelType, params: Optional[Dict]):
        if model_type == ModelType.regression:
            model = LinearRegression()
        elif model_type == ModelType.classification:
            model = LogisticRegression()
        else:
            raise InvalidModelTypeError(model_type.value)

        if params is not None:
            model.set_params(**params)

        return model

    def _load_to_inference(self, model_id: str, model_type: ModelType, params: Optional[Dict]):
        model_config = self.model_bank[model_id]

        if model_type != model_config.model_type:
            raise MismatchModelTypeError(model_type.value)

        model = self._get_model_by_type(model_type, params)

        model_config.model = model

        self.inference_models[model_id] = model_config
        del self.model_bank[model_id]
    #
    # def unload(self):
    #     self.inference_model_id = None
    #
    # def get_status(self):
    #     if self.inference_model_id:
    #         return f"Model loaded: {self.inference_model_id}"
    #     return "No model loaded"
    #
    # def predict(self, X):
    #     if not self.inference_model_id:
    #         raise ModelNotFoundError("No model loaded for inference")
    #     model = self.models.get(self.inference_model_id)
    #     return model.predict(X)



    # def remove(self, model_id):
    #     if model_id in self.models:
    #         del self.models[model_id]
    #     else:
    #         raise ModelNotFoundError(model_id)
    #
    # def remove_all(self):
    #     self.models.clear()

model_manager = ModelManager(
    settings.models_path,
    settings.num_cores,
    settings.max_inference_models
)