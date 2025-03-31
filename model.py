from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import pickle as pkl

path = r"/Users/jakubklas/Documents/Code/ml_project/data.csv"
data = pd.read_csv(path)[["height", "weigth", "shoe_size"]]

X = data.drop("shoe_size", axis = 1)
y = data["shoe_size"]

model = LinearRegression()
model.fit(X, y)

def predict_value(value: float) -> float:
    return model.predict(np.array([[value]]))[0]


"""
with open(path, "w") as file:
    pickle.save(model)
"""