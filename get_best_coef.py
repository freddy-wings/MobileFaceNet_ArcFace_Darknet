import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import PolynomialFeatures

with open('cosine_score.txt', 'r') as f:
    lists = f.readlines()

def map_func(x):
    x = x.strip().split(' ')
    x = list(map(float, x))
    return x

## read file
lists = np.array(list(map(map_func, lists)))
gt, pred = np.hsplit(lists[:, 1:], indices_or_sections=2)

## get `pred != nan` or `pred != inf`
idx = np.bitwise_not(np.bitwise_or(np.isnan(pred), np.isinf(pred)))
gt = gt[idx]; pred = pred[idx]

plt.figure(0)
plt.scatter(np.arange(pred.shape[0]), pred, c=gt)
plt.show()

y_true = gt.reshape(-1, 1)
X = pred.reshape(-1, 1)
# X = np.r_[pred, np.ones(pred.shape[0])].reshape(2, -1).T

## logistic regression
model = LogisticRegression()
model.fit(X, y_true)
y_pred = model.predict(X)
acc = accuracy_score(y_true, y_pred)

print("accuracy", acc)
print("coef_", model.coef_)
print("intercept_", model.intercept_)