import os
import xgboost as xgb
import requests

from urllib.parse import urlparse
from sklearn.datasets import load_svmlight_file


TRAIN_DATASET_URL = "https://raw.githubusercontent.com/dmlc/xgboost/master/demo/data/agaricus.txt.train"
TEST_DATASET_URL = "https://raw.githubusercontent.com/dmlc/xgboost/master/demo/data/agaricus.txt.test"


def _download_file(url: str) -> str:
    parsed = urlparse(url)
    file_name = os.path.basename(parsed.path)
    file_path = os.path.join(os.getcwd(), "mushroom-xgboost", file_name)

    response = requests.get(url)

    with open(file_path, "wb") as file:
        file.write(response.content)

    return file_path


train_dataset_path = _download_file(TRAIN_DATASET_URL)
test_dataset_path = _download_file(TEST_DATASET_URL)

X_train, y_train = load_svmlight_file(train_dataset_path)
X_test, y_test = load_svmlight_file(test_dataset_path)
X_train = X_train.toarray()
X_test = X_test.toarray()

dtrain = xgb.DMatrix(data=X_train, label=y_train)

param = {"max_depth": 2, "eta": 1, "objective": "binary:logistic"}
num_round = 2
bst = xgb.train(param, dtrain, num_round)
# print(bst)

model_file_name = "./mushroom-xgboost/mushroom-xgboost.json"
bst.save_model(model_file_name)
