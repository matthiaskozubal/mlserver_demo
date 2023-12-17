import os
import pandas as pd
from dotenv import load_dotenv
from entsoe import EntsoePandasClient

load_dotenv()
entsoe_api_key = os.getenv("ENTSOE_API_KEY")

start = pd.Timestamp("20220601", tz="Europe/Brussels")
end = pd.Timestamp("20231201", tz="Europe/Brussels")
country_code = "UK"
output_file_name = "entose_train_data_load.json"

client = EntsoePandasClient(api_key=entsoe_api_key)
df = client.query_load(country_code, start=start, end=end)
with open(output_file_name, "w") as file:
    file.write(df.to_json())
