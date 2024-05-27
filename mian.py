# Importing 
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

""" --- DATA PHASE --- """

''' Data Exploration '''


# Loading The Dataset
data = pd.read_csv("h_data.csv")

# To see if there is null attributs
#print(data.info())

# Drop the rows that has empty attributs "inplace=True": To apply changes to the dataset
data.dropna(inplace=True)

# Separating the  attributes and resaults (target)
x = data.drop(["target"], axis = 1)
y = data["target"]

#print(x)

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
'''
train_data = x_train.join(y_train)

# Visualisation the data set
print(train_data.hist(figsize=(15,8)))
plt.figure(figsize=(15,8))
print(sns.heatmap(train_data.corr(), annot=True, cmap="YlGnBu"))
'''
# Tesors 
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

x_train = torch.from_numpy(x_train.astype(np.float32))
x_test = torch.from_numpy(x_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

# Tensor shaping

y_train = y_train.view(y_train.shape[0],1)
y_test = y_test.view(y_test.shape[0],1)

''' MODEL '''
# f(x) = wx + b, sigmoid at the end

class LogisticRegression(nn.Module):
    
    def __init__ (self, nbInputAtt):
        super(LogisticRegression,self).__init__()
        self.linear = nn.Linear(nbInputAtt, 1)

    def forward(self,x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted
    
model = LogisticRegression(11)

# Loss and optimizer
# Note: This func compare the resault with the real output


crit = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(),lr = 0.01)
# LR = learning rate

# Training loop
nbEpochs = 100
for epoch in range(nbEpochs):
    # Froward pass and loss
    y_predicted = model(x_train)
    loss = crit(y_predicted,y_train)
    
    # backward pass
    loss.backward()
    
    # updates
    optimizer.step()
    
    # EMPTY GRADS
    optimizer.zero_grad() 

    if(epoch+1) %10 ==0:
        print(f'epoch: {epoch+1}, loss= {loss.item():.4f}')


""" Testing Phase """
# accuracy

with torch.no_grad():
    y_predicted = model(x_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f"accuracy = {acc:.4f}")
