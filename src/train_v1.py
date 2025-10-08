import sys

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression


def train_model(X_train_path, y_train_path):
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train.values.ravel())
    return model


if __name__ == "__main__":
    X_train_path = sys.argv[1]
    y_train_path = sys.argv[2]
    model_path = sys.argv[3]

    model = train_model(X_train_path, y_train_path)
    joblib.dump(model, model_path)
