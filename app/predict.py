import joblib
import numpy as np
from sklearn.datasets import load_iris
from app.utils import get_logger

logger = get_logger(__name__)
# Load the model
model = joblib.load("app/model/iris_model.joblib")
logger.info("Model loaded successfully")
class_names = load_iris().target_names
logger.info("Class names loaded successfully")

def predict(input_data: list) -> dict:
    """
    Predict the class of the input data using the pre-trained
    machine learning model.
    Args:
        input_data (list): A list of input data points.
    Returns:
        dict: A dictionary containing the predicted class and
              the corresponding class name.
    """
    logger.debug(f"Input data: {input_data}")
    # Convert input data to numpy array
    input_data = np.array(input_data).reshape(1, -1)
    logger.debug(f"Reshaped input data: {input_data}")
    
    # Make prediction
    prediction = model.predict(input_data)
    logger.debug(f"Prediction: {prediction}")
    
    # Get class name
    class_name = class_names[prediction[0]]
    
    return {
        "predicted_class": int(prediction[0]),
        "class_name": class_name
    }
