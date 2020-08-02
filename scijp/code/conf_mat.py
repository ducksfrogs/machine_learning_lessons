import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)

from sklearn import linear_model
clf = linear_model.LogisticRegression()

clf.fit(X_train, y_train)
clf.score(X_test, y_test)

np.count_nonzero(y_test==0), np.count_nonzero(y_test==1)

y_pred = clf.predict(X_test)

conf_mat = np.zeros([2,2])

for true_label, est_label in zip(y_test, y_pred):
    conf_mat[true_label, est_label] +=1

import pandas as pd
df = pd.DataFrame(conf_mat, columns=['pred 0', 'pred 1'], index=['true 0', 'true 1'])

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

accuracy_score(y_test, y_pred)

cmat = confusion_matrix(y_test,y_pred)

TP = cmat[0,0]
TN = cmat[1,1]
FP = cmat[1,0]
FN = cmat[0,1]

#multi class

from sklearn.datasets import load_digits
data = load_digits()

X = data.data
y = data.target

img = data.images

import matplotlib.pyplot as plt

plt.gray()
plt.imshow(img[0], interpolation='none')
plt.axis('off')

for i in range(10):
    i_th_digit = data.images[data.target == i]
    for j in range(0,15):
        plt.subplot(10,15, i*15 + j +1)
        plt.axis('off')
        plt.imshow(i_th_digit[j], interpolation='none')
