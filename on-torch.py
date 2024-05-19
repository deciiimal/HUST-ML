import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch import optim
from torch import cuda
from torch.utils.data import DataLoader

from torchvision import models


os.chdir('/home/HUANG/Desktop/MLPj')
device = torch.device('cuda') if cuda.is_available() else torch.device('cpu')

class Dataset:
    def __init__(self, X, y=None, test=False):
        self.X = X
        self.y = y
        self.test = test

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        X = self.X[idx].reshape(-1, 28, 28).astype(np.float32)
        if self.test:
            return torch.tensor(X)
        y = self.y[idx]
        return torch.tensor(X), torch.tensor(y)


train_csv = pd.read_csv('data/train.csv')
test_csv = pd.read_csv('data/test.csv')

X = train_csv.iloc[:, 1:].to_numpy()
y = train_csv.iloc[:, 0].to_numpy()
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
X_test = test_csv.to_numpy()
train_dataset = Dataset(X_train, y_train)
valid_dataset = Dataset(X_valid, y_valid)
test_dataset = Dataset(X_test, test=True)

train_data = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
valid_data = DataLoader(valid_dataset, batch_size=128, shuffle=True, num_workers=4)
test_data = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
net.fc = nn.Linear(net.fc.in_features, 10)
optimizer = optim.Adam(net.parameters(), lr=0.002)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
criterion = nn.CrossEntropyLoss()


def train_(epochs, train_data, valid_data, device):
    net.to(device)
    for epoch in range(epochs):
        correct, total, loss = 0, 0, None
        train_acc, valid_acc = 0, 0
        net.train()
        for i, (X, y) in enumerate(train_data):

            X: torch.Tensor
            y: torch.Tensor

            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = net(X)
            print(y.shape, y_hat.shape)
            exit(0)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            total += y.size(0)
            correct += (y_hat.argmax(dim=1) == y).sum()
        train_acc = correct / total

        correct, total = 0, 0
        net.eval()
        with torch.no_grad():
            for X, y in valid_data:

                X: torch.Tensor
                y: torch.Tensor

                X, y = X.to(device), y.to(device)
                y_hat = net(X)
                total += y.size(0)
                correct += (y_hat.argmax(dim=1) == y).sum()
        valid_acc = correct / total

        print(f'Train epoch {epoch+1}: Loss: {loss.mean():.4f}, Train acc: {train_acc:.4f}, Valid acc: {valid_acc:.4f}')


epoch = 100
train_(epoch, train_data, valid_data, device)
# torch.save(net.state_dict(), 'resnet50.params')

# net.load_state_dict(torch.load('resnet50.params'))
# net.to(device)
# submission = pd.DataFrame(columns=['ImageId', 'Label'])
# print(submission)
# label = []
# net.eval()
# with torch.no_grad():
#     for X in test_data:
#         X: torch.Tensor
#         y_hat: torch.Tensor = net(X.to(device)).argmax(dim=1)
#         label.extend(y_hat.cpu().tolist())
# submission['Label'] = np.array(label)
# submission['ImageId'] = np.arange(1, len(test_dataset) + 1)
# submission.to_csv('submission.csv', index=False)
