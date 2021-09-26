import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn.model_selection import learning_curve

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import KFold

from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPRegressor

from sklearn import svm

import time


nycSales_data = pd.read_csv('/Users/Aaron/Desktop/nyc-rolling-sales.csv')
nycSales_data = nycSales_data.replace(' -  ', 0)
nycSales_data['SALE PRICE'] = pd.to_numeric(nycSales_data['SALE PRICE'])
nycSales_data['YEAR BUILT'] = pd.to_numeric(nycSales_data['YEAR BUILT'])
nycSales_data['LAND SQUARE FEET'] = pd.to_numeric(nycSales_data['LAND SQUARE FEET'])
nycSales_data['GROSS SQUARE FEET'] = pd.to_numeric(nycSales_data['GROSS SQUARE FEET'])
nycSales_data.drop(nycSales_data[nycSales_data['SALE PRICE'] < 500000].index, inplace=True)
nycSales_data.drop(nycSales_data[nycSales_data['LAND SQUARE FEET'] == 0].index, inplace=True)
nycSales_data.drop(nycSales_data[nycSales_data['GROSS SQUARE FEET'] == 0].index, inplace=True)
nycSales_data.drop(nycSales_data[nycSales_data['YEAR BUILT'] == 0].index, inplace=True)

x1 = nycSales_data
N1 = set(x1['NEIGHBORHOOD'])
n_to_id = dict(zip(N1, range(len(N1))))
x1['NEIGHBORHOOD'] = x1['NEIGHBORHOOD'].map(n_to_id)
id_to_n = list(N1)

nycSales_data.to_csv(r'/Users/Aaron/Desktop/data_cleaned.csv')

X = nycSales_data[['BOROUGH', 'NEIGHBORHOOD', 'LAND SQUARE FEET', 'ZIP CODE', 'GROSS SQUARE FEET', 'YEAR BUILT', 'TAX CLASS AT TIME OF SALE']]
# X = nycSales_data[['BOROUGH', 'NEIGHBORHOOD', 'BUILDING CLASS CATEGORY', 'BUILDING CLASS AT TIME OF SALE', 'LAND SQUARE FEET', 'ZIP CODE', 'RESIDENTIAL UNITS', 'COMMERCIAL UNITS', 'GROSS SQUARE FEET', 'YEAR BUILT', 'TAX CLASS AT TIME OF SALE']]
X = X.astype(str).astype(int)
y = nycSales_data['SALE PRICE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8)

# Decision Tree
# model = tree.DecisionTreeRegressor()
# model = tree.DecisionTreeRegressor(max_depth=20, min_samples_split=10, min_samples_leaf=10)
model = tree.DecisionTreeRegressor(max_depth=3)
t0 = time.time()
model = model.fit(X_train, y_train)
t1 = time.time()
dt_time = t1 - t0
print('DT training: %f seconds' % dt_time)
predictions = model.predict(X_test)
t2 = time.time()
dt_time_2 = t2 - t0
print('Complete DT Prediction: %f seconds' % dt_time_2)
score = explained_variance_score(y_test, predictions)
print(score)


depth_range = np.arange(50)+1
train_scores, test_scores = validation_curve(model, X_train, y_train, param_name="max_depth", param_range=depth_range, scoring='explained_variance')

plt.figure()
plt.plot(depth_range, np.mean(train_scores, axis=1), label='Training score')
plt.plot(depth_range, np.mean(test_scores, axis=1), label='Cross-validation score')
plt.title('Validation curve for decision tree')
plt.xlabel('max_depth')
plt.ylabel("Classification score")
plt.legend(loc="best")
plt.grid()
plt.savefig('Figure 1')

train_sizes = np.linspace(0.05, 1.0, 20)
_, train_scores, test_scores = learning_curve(model, X_train, y_train, train_sizes=train_sizes)

plt.figure()
plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training score')
plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label='Cross-validation score')
plt.title('Learning curve')
plt.xlabel('Training Data Fraction')
plt.ylabel("Classification score")
plt.legend(loc="best")
plt.grid()
plt.savefig('Figure 2')

# KNN
model = KNeighborsRegressor(n_neighbors=10)
t0 = time.time()
model = model.fit(X_train, y_train)
t1 = time.time()
knn_time = t1 - t0
print('KNN training: %f seconds' % knn_time)
predictions = model.predict(X_test)
t2 = time.time()
knn_time_2 = t2 - t0
print('Complete KNN Prediction: %f seconds' % knn_time_2)
score = explained_variance_score(y_test, predictions)
print(score)

k_range = np.arange(1, 50)
train_scores, test_scores = validation_curve(model, X_train, y_train, param_name="n_neighbors", param_range=k_range, scoring='explained_variance')

plt.figure()
plt.plot(k_range, np.mean(train_scores, axis=1), label='Training score')
plt.plot(k_range, np.mean(test_scores, axis=1), label='Cross-validation score')
plt.title('Validation curve for kNN')
plt.xlabel('k')
plt.ylabel("Classification score")
plt.legend(loc="best")
plt.grid()
plt.savefig('Figure 3')

train_sizes = np.linspace(0.05, 1.0, 20)
_, train_scores, test_scores = learning_curve(model, X_train, y_train, train_sizes=train_sizes)

plt.figure()
plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training score')
plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label='Cross-validation score')
plt.title('Learning curve')
plt.xlabel('Training Data Fraction')
plt.ylabel("Classification score")
plt.legend(loc="best")
plt.grid()
plt.savefig('Figure 4')

# Boosting
dt = tree.DecisionTreeRegressor(max_depth=3)
model = AdaBoostRegressor(base_estimator=dt)
t0 = time.time()
model = model.fit(X_train, y_train)
t1 = time.time()
boost_time = t1 - t0
print('BOOST training: %f seconds' % boost_time)
predictions = model.predict(X_test)
t2 = time.time()
boost_time_2 = t2 - t0
print('Complete BOOST Prediction: %f seconds' % boost_time_2)
score = explained_variance_score(y_test, predictions)
print(score)

k_range = np.arange(1, 20) + 1
train_scores, test_scores = validation_curve(model, X_train, y_train, param_name="n_estimators", param_range=k_range, scoring='explained_variance')

plt.figure()
plt.plot(k_range, np.mean(train_scores, axis=1), label='Training score')
plt.plot(k_range, np.mean(test_scores, axis=1), label='Cross-validation score')
plt.title('Validation curve for AdaBoost')
plt.xlabel('k')
plt.ylabel("Classification score")
plt.legend(loc="best")
plt.grid()
plt.savefig('Figure 5')

train_sizes = np.linspace(0.05, 1.0, 20)
_, train_scores, test_scores = learning_curve(model, X_train, y_train, train_sizes=train_sizes)

plt.figure()
plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training score')
plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label='Cross-validation score')
plt.title('Learning curve')
plt.xlabel('Training Data Fraction')
plt.ylabel("Classification score")
plt.legend(loc="best")
plt.grid()
plt.savefig('Figure 6')

# Neural Network
model = MLPRegressor()
t0 = time.time()
model = model.fit(X_train, y_train)
t1 = time.time()
nn_time = t1 - t0
print('Neural Network training: %f seconds' % nn_time)
predictions = model.predict(X_test)
t2 = time.time()
nn_time_2 = t2 - t0
print('Complete Neural Network Prediction: %f seconds' % nn_time_2)
score = explained_variance_score(y_test, predictions)
print(score)

alpha = np.linspace(0.0001, 0.01, 20)
train_scores, test_scores = validation_curve(model, X_train, y_train, param_name="alpha", param_range=alpha, scoring='explained_variance')

plt.figure()
plt.plot(alpha, np.mean(train_scores, axis=1), label='Training score')
plt.plot(alpha, np.mean(test_scores, axis=1), label='Cross-validation score')
plt.title('Validation curve for Neural Network')
plt.xlabel('alpha')
plt.ylabel("Classification score")
plt.legend(loc="best")
plt.grid()
plt.savefig('Figure 7')

LR = np.linspace(0.001, 0.01, 20)
train_scores, test_scores = validation_curve(model, X_train, y_train, param_name="learning_rate_init", param_range=LR, scoring='explained_variance')

plt.figure()
plt.plot(LR, np.mean(train_scores, axis=1), label='Training score')
plt.plot(LR, np.mean(test_scores, axis=1), label='Cross-validation score')
plt.title('Validation curve for Neural Network')
plt.xlabel('LR')
plt.ylabel("Classification score")
plt.legend(loc="best")
plt.grid()
plt.savefig('Figure 8')

train_sizes = np.linspace(0.05, 1.0, 20)
_, train_scores, test_scores = learning_curve(model, X_train, y_train, train_sizes=train_sizes)

plt.figure()
plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training score')
plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label='Cross-validation score')
plt.title('Learning curve')
plt.xlabel('Training Data Fraction')
plt.ylabel("Classification score")
plt.legend(loc="best")
plt.grid()
plt.savefig('Figure 9')

# SVM
model = svm.LinearSVR()
t0 = time.time()
model = model.fit(X_train, y_train)
t1 = time.time()
svm_time = t1 - t0
print('SVM training: %f seconds' % svm_time)
predictions = model.predict(X_test)
t2 = time.time()
svm_time_2 = t2 - t0
print('Complete SVM Prediction: %f seconds' % svm_time_2)
score = explained_variance_score(y_test, predictions)
print(score)

C = np.arange(10)+1
train_scores, test_scores = validation_curve(model, X_train, y_train, param_name="C", param_range=C, scoring='explained_variance')

plt.figure()
plt.plot(C, np.mean(train_scores, axis=1), label='Training score')
plt.plot(C, np.mean(test_scores, axis=1), label='Cross-validation score')
plt.title('Validation curve for SVM_varying C')
plt.xlabel('k')
plt.ylabel("Classification score")
plt.legend(loc="best")
plt.grid()
plt.savefig('Figure 10')

E = np.linspace(0, 1, 20)
train_scores, test_scores = validation_curve(model, X_train, y_train, param_name="epsilon", param_range=E, scoring='explained_variance')

plt.figure()
plt.plot(E, np.mean(train_scores, axis=1), label='Training score')
plt.plot(E, np.mean(test_scores, axis=1), label='Cross-validation score')
plt.title('Validation curve for SVM_varying Epsilon')
plt.xlabel('k')
plt.ylabel("Classification score")
plt.legend(loc="best")
plt.grid()
plt.savefig('Figure 11')

train_sizes = np.linspace(0.05, 1.0, 20)
_, train_scores, test_scores = learning_curve(model, X_train, y_train, train_sizes=train_sizes)

plt.figure()
plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training score')
plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label='Cross-validation score')
plt.title('Learning curve')
plt.xlabel('Training Data Fraction')
plt.ylabel("Classification score")
plt.legend(loc="best")
plt.grid()
plt.savefig('Figure 12')