import requests
import numpy as np

from mlserver.types import InferenceRequest
from mlserver.codecs import NumpyCodec


x_0 = np.array([28.0])
inference_request = InferenceRequest(
    inputs=[NumpyCodec.encode_input(name="marriage", payload=x_0)]
)
print(f"{inference_request = }\n")
print(f"{inference_request.dict() = }\n")
endpoint = "http://localhost:8080/v2/models/numpyro-divorce/infer"
response = requests.post(endpoint, json=inference_request.dict())
print(f"{response.json() = }\n")
