import pandas as pd
import joblib
import numpy as np

# Загружаем предобученную модель
model = joblib.load("trained_model.pkl")
test_data = pd.read_csv("test/test_data.csv")

test_data['random_feature'] = np.random.randn(len(test_data))

predictions = model.predict(test_data.drop(columns="temperature"))
print("Предсказанные значения:")
print(predictions)