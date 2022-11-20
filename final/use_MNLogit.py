import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from requests import session
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, accuracy_score)
import scipy.stats as stats
import statsmodels.discrete.discrete_model as sm

from load_data1 import loadData1
from load_data2 import loadData2
labels, features, feature_names, num_labels = loadData2()
print("labels: ", labels.shape)
print("features: ", features.shape)
# print(feature_names)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=1)
useSMOTE = True

if(useSMOTE):
    os = SMOTE(random_state=0)
    os_data_X,os_data_y=os.fit_resample(X_train, y_train)
    X_train = pd.DataFrame(data=os_data_X)
    y_train = pd.DataFrame(data=os_data_y)
    # print("length of oversampled data is ",len(os_data_X))
    # print(y_train[0].value_counts())

y_train = np.array(y_train)

num_features = features.shape[1]
print('num_features: ', num_features)

def to_onehot(y):
    data = np.zeros((num_labels))
    data[int(y)] = 1
    return data

X_train = np.reshape(X_train, (-1, num_features))
X_test = np.reshape(X_test, (-1, num_features))
y_train1 = y_train
y_train = np.array([to_onehot(y) for y in y_train])
# print("y_train.shape: ", y_train)
# y_test = np.array([to_onehot(y) for y in y_test])
# print("y_test.shape: ", y_test)

# model = sm.MNLogit(y_train, X_train, ).fit(method='nm', maxiter=20000, full_output=True, disp=True, xtol=1e-5, retall=True)
model = sm.MNLogit(y_train, X_train, ).fit(method='cg', maxiter=10000, gtol=1e-5, full_output=True, disp=True, retall=True)

summary = model.summary(xname=feature_names, title="Trigger Analysis of Migraine Groups")
print(summary)
# print(summary.as_latex())

prediction = model.predict(X_test)

prediction = np.argmax(prediction, axis=1)
idx = 0
# for p in prediction:
#     print(p, ", ", y_test[idx])
#     idx = idx+1

# # accuracy score of the model
print('Test accuracy = ', accuracy_score(y_test, prediction))