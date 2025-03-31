from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app.predict import predict
from app.utils import get_logger

logger = get_logger(__name__)
app = FastAPI(title="Iris Flower Prediction API", version="1.0.0")

class PredictionRequest(BaseModel):
    """
    Request model for prediction endpoint.
    """
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.post("/predict", response_model=dict)
def predict(request: PredictionRequest):
    try:
        logger.info(f"Received request: {request}")
        input_data = [
            request.sepal_length,
            request.sepal_width,
            request.petal_length,
            request.petal_width
        ]
        logger.debug(f"Input data for prediction: {input_data}")
        prediction = predict(input_data)
        logger.debug(f"Prediction result: {prediction}")
        return JSONResponse(content=prediction)
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error"}
        )

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response: {response.status_code}")
    return response