from starlette.requests import Request
from starlette.responses import JSONResponse


class Error(Exception):
    def __init__(self, error_code: int, message: str):
        self.error_code = error_code
        self.message = message
        super().__init__(self.message)


class ModelNotFoundError(Error):
    def __init__(self, model_id: str, error_code: int = 404):
        message = f"Model with ID '{model_id}' not exists."
        super().__init__(error_code, message)


class ModelAlreadyExistsError(Error):
    def __init__(self, model_id: str, error_code: int = 400):
        message = f"Model with ID '{model_id}' already exists."
        super().__init__(error_code, message)


class ModelAlreadyLoadedError(Error):
    def __init__(self, model_id: str, error_code: int = 400):
        message = f"Model with ID '{model_id}' already loaded to inference."
        super().__init__(error_code, message)


class NoFreeCoresError(Error):
    def __init__(self, error_code: int = 503):
        message = "No free cores available."
        super().__init__(error_code, message)


class MaxModelsReachedError(Error):
    def __init__(self, error_code: int = 503):
        message = "Maximum number of models loaded for inference reached."
        super().__init__(error_code, message)


class InvalidModelTypeError(Error):
    def __init__(self, model_type: str, error_code: int = 400):
        message = f"Invalid model type: '{model_type}'."
        super().__init__(error_code, message)

class MismatchModelTypeError(Error):
    def __init__(self, model_type: str, error_code: int = 400):
        message = f"Requested model doesn't have model type: '{model_type}'."
        super().__init__(error_code, message)


async def error_handler(request: Request, e: Error):
    return JSONResponse(status_code=e.error_code, content={"detial": str(e)})
