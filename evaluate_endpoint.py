import joblib
import requests
import numpy as np
from sklearn.metrics import classification_report
from tqdm import tqdm

# Load the test data
X_test, y_test = joblib.load("app/model/test_data.joblib")

# endpoint URL
url = "http://localhost:8000/predict"

class_names = ["setosa", "versicolor", "virginica"]

pred_labels = []

print("Sending requests to the endpoint...")
for features in tqdm(X_test):
    # Prepare the payload
    data = {
        "features": features.tolist()
    }
    
    # Send the request to the endpoint
    response = requests.post(url, json=data)
    
    # Check if the request was successful
    if response.status_code == 200:
        pred_labels.append(response.json()["pred_label"])
    else:
        print(f"Error: {response.status_code} - {response.text}")
        pred_labels.append("unknown")

# Map the predicted labels to class names
true_labels = [class_names[int(label)] for label in y_test]
valid_indices = [i for i, label in enumerate(pred_labels) if label != "unknown"]

filtered_preds = [pred_labels[i] for i in valid_indices]
filtered_true = [true_labels[i] for i in valid_indices]

# Print the classification report
print("\nClassification Report:")
print(classification_report(filtered_true, filtered_preds, target_names=class_names))