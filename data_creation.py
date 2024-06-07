import os
import numpy as np
import pandas as pd

os.makedirs("train", exist_ok=True)
os.makedirs("test", exist_ok=True)

train_data = pd.DataFrame({"temperature": np.random.normal(loc=20, scale=5, size=100)})
train_data.to_csv("train/train_data.csv", index=False)

test_data = pd.DataFrame({"temperature": np.random.normal(loc=25, scale=6, size=50)})
test_data.to_csv("test/test_data.csv", index=False)