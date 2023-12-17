import pandas as pd
from mlserver.codecs import PandasCodec

# Encode request
## DataFrame
df = pd.DataFrame({"Name": ["Alice", "Bob"], "Age": [33, 35]})
inference_request = PandasCodec.encode_request(df)
print(inference_request)
