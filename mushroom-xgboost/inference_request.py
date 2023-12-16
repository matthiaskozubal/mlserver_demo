import os
import requests
from sklearn.datasets import load_svmlight_file


test_dataset_path = os.path.join(os.getcwd(), "agaricus.txt.test")
X_test, y_test = load_svmlight_file(test_dataset_path)
X_test = X_test.toarray()
x_0 = X_test[0:1]
y_target = y_test[0]

inference_request = {
    "inputs": [
        {
            "name": "predict",
            "shape": x_0.shape,
            "datatype": "FP32",
            "data": x_0.tolist(),
        }
    ]
}

endpoint = (
    "http://localhost:8080/v2/models/mushroom-xgboost/versions/v0.1.0/infer"
)
response = requests.post(endpoint, json=inference_request)
y_pred = response.json()["outputs"][0]["data"][0]
print(f"{y_pred = }")
print(f"{y_target = }")
