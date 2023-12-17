import requests
import numpy as np
import pickle

from mlserver.types import InferenceRequest
from mlserver.codecs import NumpyCodec

# Load

X_test_file = "X_test.pkl"
y_test_file = "y_test.pkl"
X_test = pickle.load(open(X_test_file, "rb"))
y_test = pickle.load(open(y_test_file, "rb"))
X_0 = X_test[0]
y_0 = y_test[0]

inference_request = InferenceRequest(
    inputs=[NumpyCodec.encode_input(name="marriage", payload=X_0)]
)
print(f"{inference_request = }\n")
print(f"{inference_request.dict() = }\n")
endpoint = "http://localhost:8080/v2/models/xgboost-model/infer"
response = requests.post(endpoint, json=inference_request.dict())
print(f"{response.json() = }\n")
