# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 18:52:25 2023

@author: fdcel
"""


from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

def train_classifier(X_train, y_train, X_test, y_test, algorithm='random_forest', cv=5):
    if algorithm == 'logistic_regression':
        clf = LogisticRegression()
    elif algorithm == 'random_forest':
        clf = RandomForestClassifier(n_estimators=100, random_state=0)
    elif algorithm == 'knn':
        clf = KNeighborsClassifier()
    elif algorithm == 'kmeans':
        clf = KMeans(n_clusters=2)
    elif algorithm == 'svm':
        clf = SVC()
    elif algorithm == 'decision_tree':
        clf = DecisionTreeClassifier()
    elif algorithm == 'naive_bayes':
        clf = GaussianNB()
    elif algorithm == 'adaboost':
        clf = AdaBoostClassifier(n_estimators=100)
    elif algorithm == 'gradient_boosting':
        clf = GradientBoostingClassifier(n_estimators=100)
    elif algorithm == 'xgb':
        clf = XGBClassifier(n_estimators=100)
    elif algorithm == 'lgbm':
        clf = LGBMClassifier(n_estimators=100)
    elif algorithm == 'multinomial_nb':
        clf = MultinomialNB()
    elif algorithm == 'neural_network':
        model = Sequential()
        model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=10, batch_size=32)
        clf = model
    else:
        raise ValueError('Invalid algorithm specified')
    if algorithm != 'neural_network':
        scores = cross_val_score(clf, X_train, y_train, cv=cv)
        clf.fit(X_train, y_train)
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)
        accuracy = accuracy_score(y_train, y_pred_train)
        precision = precision_score(y_train, y_pred_train)
        recall = recall_score(y_train, y_pred_train)
        plt.plot(range(len(y_train)), y_train, 'bo', label='True')
        plt.plot(range(len(y_train)), y_pred_train, 'ro', label='Predicted')
        plt.legend()
        plt.show()
        plt.plot(range(len(y_test)), y_test, 'bo', label='True')
        plt.plot(range(len(y_test)), y_pred_test, 'ro', label='Predicted')
        plt.legend()
        plt.show()
        
        cm = confusion_matrix(y_test, y_pred_test)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion matrix Test')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['0', '1'], rotation=45)
        plt.yticks(tick_marks, ['0', '1'])
        plt.tight_layout()
        plt.ylabel('True label Test')
        plt.xlabel('Predicted label Test')
        plt.text(0, 0, f'TN: {cm[0, 0]}', ha='center', va='center', color='white')
        plt.text(0, 1, f'FN: {cm[1, 0]}', ha='center', va='center', color='white')
        plt.text(1, 0, f'FP: {cm[0, 1]}', ha='center', va='center', color='white')
        plt.text(1, 1, f'TP: {cm[1, 1]}', ha='center', va='center', color='white')
        plt.show()
        
        cm = confusion_matrix(y_train, y_pred_train)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion matrix Train')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['0', '1'], rotation=45)
        plt.yticks(tick_marks, ['0', '1'])
        plt.tight_layout()
        plt.ylabel('True label Train')
        plt.xlabel('Predicted label Train')
        plt.text(0, 0, f'TN: {cm[0, 0]}', ha='center', va='center', color='white')
        plt.text(0, 1, f'FN: {cm[1, 0]}', ha='center', va='center', color='white')
        plt.text(1, 0, f'FP: {cm[0, 1]}', ha='center', va='center', color='white')
        plt.text(1, 1, f'TP: {cm[1, 1]}', ha='center', va='center', color='white')
        plt.show()
    else:
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)
        accuracy = accuracy_score(y_train, y_pred_train)
        precision = precision_score(y_train, y_pred_train)
        recall = recall_score(y_train, y_pred_train)
        
    print("accuracy",accuracy)
    print("precision",precision)
    print("recall",recall)
    return clf,scores, accuracy, precision, recall



import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the Titanic dataset
df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

# Preprocess the data
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
df = df.fillna(df.mean())
df = pd.get_dummies(df, columns=['Sex', 'Embarked'])

# Split the data into X and y
X = df.drop('Survived', axis=1)
y = df['Survived']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a logistic regression classifier
clf,scores, accuracy, precision, recall=train_classifier(X_train, y_train, X_test, y_test, algorithm='logistic_regression')
