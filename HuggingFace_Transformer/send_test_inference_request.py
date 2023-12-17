import requests


inference_request = {
    "inputs": [
        {
            "name": "args",
            "shape": [1],
            "datatype": "BYTES",
            "data": ["I am using MLServer to serve the model."],
        }
    ]
}

response = requests.post(
    "http://localhost:8080/v2/models/transformer/infer",
    json=inference_request,
)
print(response.json())
