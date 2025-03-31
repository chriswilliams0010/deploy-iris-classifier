from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train a Support Vector Classifier
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)
# Save the model to a file
joblib.dump(model, "app/model/iris_model.joblib")
joblib.dump([X_test, y_test], "app/model/test_data.joblib")
print("Model trained and saved successfully.")
