#!/usr/bin/env python
# coding: utf-8

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,  TensorDataset
import torch.nn.functional as F
import torch.nn as nn
import torch
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataframe = pd.read_csv('./Churn.csv')
# 7044 rows, 6575 cols
x = pd.get_dummies(dataframe.drop(['Churn', 'Customer ID'], axis=1))
y = dataframe['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=33)
X_train = torch.FloatTensor(x_train.values)  # [5635, 6575]
X_test = torch.FloatTensor(x_test.values)
Y_train = torch.LongTensor(y_train.values)  # [5635]
Y_test = torch.LongTensor(y_test.values)
train_combined = TensorDataset(X_train, Y_train)
test_combined = TensorDataset(X_test, Y_test)
batch_size = 500
train_loader = DataLoader(train_combined, batch_size,
                          shuffle=True)
test_loader = DataLoader(test_combined, shuffle=False)


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


input_size = len(X_train[0])
output_size = 2
torch.manual_seed(32)
hidden_layer_1_nodes = 360
hidden_layer_2_nodes = 40
model = Model(in_features=input_size, h1=hidden_layer_1_nodes,
              h2=hidden_layer_2_nodes, out_features=output_size)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model).cuda()

model.to(device)
criterion = nn.CrossEntropyLoss().cuda(
) if torch.cuda.is_available() else nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 120

# per epoch
for i in range(epochs):
    i += 1
    # per batch
    val_correct_preds = 0
    count = 0
    if i % 10 == 1:
        print(f"epoch: {i} out of {epochs} completed.")
    losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        model.train()
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # training
        y_pred = model.forward(data)
        loss = criterion(y_pred, target)
        losses.append(loss)
        # backtracking
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if i % 10 == 1:
        print(f"Max loss in this batch: {max(losses)}")
print("Done training!")

correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        y_val = model.forward(data)
#         print("predicted:", y_val.argmax().item(), "actual: ", target.item())
        if y_val.argmax().item() == target.item():
            correct += 1
print(f'\n{correct} out of {len(test_loader)} = {100*correct/len(test_loader):.2f}% correct')
torch.save(model.state_dict(), 'mlp.pt')
