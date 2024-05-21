import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score
from myKNN import *

train_csv = pd.read_csv('data/train.csv')
test_csv = pd.read_csv('data/test.csv')

X = train_csv.iloc[:, 1:].to_numpy(dtype=np.float32) / 256.
y = train_csv.iloc[:, 0].to_numpy()

X, _, y, _ = train_test_split(X, y, test_size=0.75)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
X_test = test_csv.to_numpy(dtype=np.float32) / 256.

print(X_train.shape, X_valid.shape, X_test.shape)

model = KNN()

model.fit(X_train, y_train)

pred = model.predict(X_valid, batch_size=128)

print(accuracy_score(y_valid, pred))
print(recall_score(y_valid, pred, average='macro'))
print(f1_score(y_valid, pred, average='macro'))

# submission = pd.DataFrame(columns=['ImageId', 'Label'])

# label = []
# pred: np.ndarray = model.predict(X_test)
# label.extend(pred.tolist())

# submission['Label'] = np.array(label)
# submission['ImageId'] = np.arange(1, len(X_test) + 1)
# submission.to_csv('submission3.csv', index=False)


