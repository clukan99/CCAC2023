### This is a simple model with one middle layer for a logistic regression neural network

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
import sklearn
from datetime import date
from datetime import datetime
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from sklearn.model_selection import train_test_split


print("Reading in the data")
########
now = datetime.now()
print("now =", now)
########
trucks = pd.read_csv("../fully_cleaned_super_nice_data.csv")
trucks = trucks.dropna()
print("Data reading is complete")
########
now = datetime.now()
print("now =", now)
########

y_data = trucks[['Sale_Price']]
X_data = trucks.drop(labels= ['Sale_Price'], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, train_size= 0.8, random_state= 440)

X_train_columns = X_train.columns
X_test_columns = X_test.columns
y_train_columns = y_train.columns
y_test_columns = y_test.columns

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

X_train = torch.tensor(X_train, dtype= torch.float32, requires_grad= True)
X_test =  torch.tensor(X_test, dtype = torch.float32, requires_grad= True)
y_train = torch.tensor(y_train, dtype = torch.float32, requires_grad= True)
y_test =  torch.tensor(y_test,dtype = torch.float32, requires_grad= True)


#import IPython; IPython.embed(); exit(1)

input_size = X_train.shape[1]
print(input_size)
hidden_layer1_size = input_size//2
hidden_layer2_size = hidden_layer1_size//2
hidden_layer3_size = hidden_layer2_size//2

output_size = 1
dropout = 0.3

########
now = datetime.now()
print("now =", now)
########

print("Starting the training now")


class LinearRegression(nn.Module):
    def __init__(self,input_dim, hidden_layer1, hidden_layer2,output_dim,p):
        super(LinearRegression, self).__init__()
        
        # Define layers
        self.linear1 = nn.Linear(input_dim, hidden_layer1)
        self.relu = nn.ReLU()

        self.linear2 = nn.Linear(in_features= hidden_layer1,out_features= output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.linear3 = nn.Linear(hidden_layer2, output_dim)
        self.relu = nn.ReLU()



        self.dropout = nn.Dropout(p = p)
        

    def forward(self,x):
        out = self.linear1(x)
        out = self.relu(out)
        #out = self.softmax(out)

        #out = self.dropout(out)

        #out = self.linear2(out)
        y_predicted = self.linear2(out)

        #out = self.relu(out)
        #out = self.dropout(out)

        #y_predicted = self.linear3(out)
        #y_predicted = self.relu(out)
        return y_predicted


model = LinearRegression(input_dim =input_size,hidden_layer1 =hidden_layer1_size ,hidden_layer2 =hidden_layer2_size, output_dim =output_size, p = dropout)


learning_rate = 0.01
n_iters = 75
loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

for epoch in range(n_iters):
    # Forward
    y_pred = model(X_train)

    # Loss
    l = loss(y_pred,y_train)

    #Gradient = backward pass
    l.backward() # dl/dw

    #Update the weights
    optimizer.step()

    # zero gradients
    optimizer.zero_grad()

    if epoch % 10 == 0:
        print(f'epoch {epoch +1}: loss = {l}')


with torch.no_grad():
    y_test_pred = model(X_test)

    
########
now = datetime.now()
print("now =", now)
########
print("Finished training; Time to save")

y_test_pred = y_test_pred.detach().numpy()
y_test_pred = y_test_pred.tolist()
y_test = y_test.detach().numpy()
y_test = y_test.tolist()
'''
dicked = {'prediction': y_test_pred.detach().numpy(), 
          'test':y_test.detach().numpy()}
'''
#import IPython; IPython.embed(); exit(1)
dicked = {'prediction': y_test_pred, 
          'test':y_test}
undicked = pd.DataFrame(dicked)

def _make_float(x):
    return float(x)

def _make_int(x):
    return int(x)
'''
undicked['test'] = undicked['test'].str.strip('[].')
undicked['prediction'] = undicked['prediction'].str.strip('[]')
undicked['prediction'] = undicked['prediction'].apply(_make_float)
undicked['test'] = undicked['test'].apply(_make_int)
'''
import IPython; IPython.embed(); exit(1)




pred = undicked['prediction'][0]
pred = pd.Series(pred)
test = undicked['test'][0]
test = pd.Series(test)





NN1 = undicked
NN1['test'] = NN1['test'].str.strip('[].')
NN1['prediction'] = NN1['prediction'].str.strip('[]')
def _make_float(x):
    return float(x)

def _make_int(x):
    return int(x)



NN1 = {'prediction': NN1['prediction'][0],
       'test': NN1['test'][0]}

#import IPython; IPython.embed(); exit(1)

NN1 = pd.DataFrame([NN1])
import IPython; IPython.embed(); exit(1)

NN1['prediction'] = NN1['prediction'].apply(_make_float)
NN1['test'] = NN1['test'].apply(_make_int)




print("Saving the data")
undicked.to_csv("../results/ANN_numerical_two_layers_cleaned_results_nodrop.csv")
import IPython; IPython.embed(); exit(1)