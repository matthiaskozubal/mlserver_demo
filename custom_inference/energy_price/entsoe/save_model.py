import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pickle
import numpy as np


# Load
data_filename = "entose_train_data_load.json"
df = pd.read_json(data_filename)

# Preprocess
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_load = scaler.fit_transform(df.values.reshape(-1, 1))

# Reshape
X = []
y = []
for i in range(1, len(scaled_load)):
    X.append(scaled_load[i - 1 : i, 0])
    y.append(scaled_load[i, 0])

X, y = np.array(X), np.array(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)
## save
pickle.dump(X_test, open("X_test.pkl", "wb"))
pickle.dump(y_test, open("y_test.pkl", "wb"))

model = xgb.XGBRegressor(
    objective="reg:squarederror",
    colsample_bytree=0.3,
    learning_rate=0.1,
    max_depth=5,
    alpha=10,
    n_estimators=100,
)

model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)

# Save
filename = "xgboost_model.pkl"
pickle.dump(model, open(filename, "wb"))
