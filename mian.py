# Importing 
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

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

print(x)

''' Data Preprocessing '''

# Separating The test and train sets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
x_train , x_test , y_train, y_test = train_test_split(x,y, test_size=0.2)

# Scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

# Creating the whole training dataset (Attributes + target)
train_data = x_train.join(y_train)

# Visualisation the data set
print(train_data.hist(figsize=(15,8)))
plt.figure(figsize=(15,8))
print(sns.heatmap(train_data.corr(), annot=True, cmap="YlGnBu"))

# Tesors 

x_train = torch.from_numpy(x_train.astype(np.float32))
x_test = torch.from_numpy(x_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

# Tensor shaping

y_train = y_train.view(y_train.shape[0],1)
y_test = y_test.view(y_test.shape[0],1)

''' MODEL '''
