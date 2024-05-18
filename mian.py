import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import sklearn

data = pd.read_csv("h_data.csv")

#print(data.info())

# data.dropna(inplace=True)
x = data.drop(["target"], axis = 1)
y = data["target"]

# print(y)
from sklearn.model_selection import train_test_split
x_train , x_test , y_train, y_test = train_test_split(x,y, test_size=0.2)

print(x_test.info())