# Importing 
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import sklearn

""" --- DATA PHASE --- """

''' Data Exploration '''


# Loading The Dataset
data = pd.read_csv("h_data.csv")

# To see if there is null attributs
print(data.info())

# Drop the rows that has empty attributs "inplace=True": To apply changes to the dataset
data.dropna(inplace=True)

# Separating the  attributes and resaults (target)
x = data.drop(["target"], axis = 1)
y = data["target"]

# Separating The test and train sets
from sklearn.model_selection import train_test_split
x_train , x_test , y_train, y_test = train_test_split(x,y, test_size=0.2)
 

# Creating the whole training dataset (Attributes + target)
train_data = x_train.join(y_train)

# Visualisation the data set
print(train_data.hist(figsize=(15,8)))
plt.figure(figsize=(15,8))
print(sns.heatmap(train_data.corr(), annot=True, cmap="YlGnBu"))


''' Data Preprocessing '''
