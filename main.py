import uvicorn
from fastapi import FastAPI
from mlapp.config import settings
from mlapp.exceptions import Error, error_handler
from mlapp.router import router

app = FastAPI()


@app.on_event("startup")
def load_settings():
    print("Application settings loaded:")
    print(f"Model Directory: {settings.models_path}")
    print(f"Number of Cores: {settings.num_cores}")
    print(f"Max Models: {settings.max_inference_models}")


app.include_router(router)
app.add_exception_handler(Error, error_handler)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
