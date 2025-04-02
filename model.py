from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import pickle

"""
path = r"/Users/jakubklas/Documents/Code/ml_project/data.csv"
data = pd.read_csv(path)
print(data)

X = data.drop("gender", axis = 1)
y = data["gender"]


model = RandomForestClassifier()
model.fit(X, y)


path = r"/Users/jakubklas/Documents/Code/ml_project/model.pkl"
with open(path, 'wb') as file:
    pickle.dump(model, file)
"""

path = r"/Users/jakubklas/Documents/Code/ml_project/model.pkl"
with open(path, 'rb') as file:
    model = pickle.load(file)


def predict_class(features: list[list[float]]) -> list[float]:
    classes = model.predict(features).tolist()
    return classes

def predict_proba(features: list[list[float]]) -> list[float]:
    probas = model.predict_proba(features).tolist()
    return probas
