import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

train_data = pd.read_csv("train/train_data_scaled.csv")

train_data['random_feature'] = np.random.randn(len(train_data))

print("Признаки в данных для обучения после добавления случайного признака:")
print(train_data.columns)

model = LinearRegression()
model.fit(train_data.drop(columns="temperature"), train_data["temperature"])

joblib.dump(model, "trained_model.pkl")