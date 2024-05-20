from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from myCNN.nn import *
from myCNN.optim import *
from myCNN.utils import *
import torch
import pandas as pd
import numpy as np

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
        return torch.tensor(X / 256.), torch.tensor(y)


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

net = Sequential(
    Conv2d(1, 24, 5, 1, 0), ReLU(),
    MaxPool2d(2, 2, 0),
    Conv2d(24, 60, 5, 1, 0), ReLU(),
    MaxPool2d(2, 2, 0),
    Flatten(),
    Linear(60*4*4, 240), Dropout(0.8), ReLU(),
    Linear(240, 84), Dropout(0.8), ReLU(),
    Linear(84, 10)
)
optimizer = SGD(lr=0.001)
criterion = CrossEntropyLoss()


def train_(epochs, train_data, valid_data):
    print('start training...')
    for epoch in range(epochs):
        if (epoch + 1) % 10 == 0:
            optimizer.lr *= 0.4
        net.train()
        correct, total, loss = 0, 0, None
        train_acc, valid_acc = 0, 0
        for i, (X, y) in enumerate(train_data):
            X, y = X.numpy(), y.numpy()
            optimizer.zero_grad(net.parameters())
            y_hat = net.forward(X)

            loss = criterion(y, y_hat)
            net.backward(criterion.grad)
            optimizer.step(net.parameters())
            total += y.shape[0]
            correct += (y_hat.argmax(axis=1) == y).sum()
        train_acc = correct / total

        net.eval()
        correct, total = 0, 0
        for X, y in valid_data:
            X, y = X.numpy(), y.numpy()
            y_hat = net.forward(X)
            total += y.shape[0]
            correct += (y_hat.argmax(axis=1) == y).sum()
        valid_acc = correct / total

        print(f'Train epoch {epoch+1}: Loss: {loss.mean()}, Train acc: {train_acc}, Valid acc: {valid_acc}')
    print('Finish training.')

check_input = np.random.randn(1, 1, 28, 28)
print(check_input.shape)
for m in net.modules():
    check_input = m.forward(check_input)
    print(f'--> {m.name} -->{check_input.shape}')

epoch = 30
train_(epoch, train_data, valid_data)

save('param.npz', net.parameters())
print('Successfully saved parameters')

# load('param.npz', net.parameters())
# print('Successfully loaded parameters')

submission = pd.DataFrame(columns=['ImageId', 'Label'])
print(submission)
label = []
net.eval()
for X in test_data:
    X = X.numpy()
    y_hat: np.ndarray = net.forward(X).argmax(axis=1)
    label.extend(y_hat.tolist())

submission['Label'] = np.array(label)
submission['ImageId'] = np.arange(1, len(test_dataset) + 1)
submission.to_csv('submission2.csv', index=False)
