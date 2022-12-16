#!/usr/bin/env python
# coding: utf-8


from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch
import pandas as pd
pd.__version__
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def explain(val, name):
    print(name + "\n")
    print(type(val))
    print(val)
    return


dataframe = pd.read_csv('./Churn.csv')
# x is the dummies of the Churn.csv, without the columns 'Churn' or 'Customer ID', or the first row?
x = pd.get_dummies(dataframe.drop(['Churn', 'Customer ID'], axis=1))
# Type DataFrame, 7044 x 6575 :  full set sans Churn/Customer ID columns.
# explain(x, 'x')
# y is the dataframe, but with Churn's values converted from strings to 1/0s
y = dataframe['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
# Type Series, 1x7044 : the true/false booleanization of the Churn column only.
# explain(y, 'y')


class Model(nn.Module):
    def __init__(self, in_features, h1, h2, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)    # input layer
        self.fc2 = nn.Linear(h1, h2)            # hidden layer
        self.out = nn.Linear(h2, out_features)  # output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x


# Splits columns,
# x_train : 80% of the x data, used to train.
# x_test : 20% of the x data, used to test
# y_train : 80% of the y column, used to train (same items as x)
# y_test : 20% of the y data, used to test.
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=33)
X_train = torch.FloatTensor(x_train.values)
X_test = torch.FloatTensor(x_test.values)
Y_train = torch.LongTensor(y_train.values)
Y_test = torch.LongTensor(y_test.values)
input_size = len(x_train.columns)
output_size = 2
torch.manual_seed(32)
hidden_layer_1_nodes = 360
hidden_layer_2_nodes = 40
model = Model(in_features=input_size, h1=hidden_layer_1_nodes,
              h2=hidden_layer_2_nodes, out_features=output_size)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

model.to(device)
trainloader = DataLoader(X_train, batch_size=30, shuffle=True)
testloader = DataLoader(X_test, batch_size=30, shuffle=False)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 120
losses = []

for i in range(epochs):
    i += 1
    y_pred = model.forward(X_train)
    loss = criterion(y_pred, Y_train)
    losses.append(loss)
    if i % 10 == 1:
        print(f'epoch: {i:2}  loss: {loss.item():10.8f}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

correct = 0
with torch.no_grad():
    for i, data in enumerate(X_test):
        y_val = model.forward(data)
#         print(f'{i+1:2}. {str(y_val):38}  {Y_test[i]}')
        if y_val.argmax().item() == Y_test[i]:
            correct += 1
print(f'\n{correct} out of {len(y_test)} = {100*correct/len(y_test):.2f}% correct')

torch.save(model.state_dict(), 'mlp.pt')
