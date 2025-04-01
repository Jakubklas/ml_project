from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import pickle as pkl

path = r"/Users/jakubklas/Documents/Code/ml_project/data.csv"
data = pd.read_csv(path)[["height", "weigth", "shoe_size", "gender"]]

X = data.drop("gender", axis = 1)
y = data["gender"]

model = RandomForestClassifier()
model.fit(X, y)
"""
with open(path, 'wb'):
    pkl.dump(model)
"""
def predict_values(features: list[list[float]]) -> list[float]:
    array = np.array(features)
    return model.predict(array)

def predict_probas(features: list[list[float]]) -> list[float]:
    array = np.array(features)
    return model.predict_proba(array)