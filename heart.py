import pandas as pd
import numpy as np
import sklearn
import flask

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import pickle


df = pd.read_csv('heart.csv')

x = df.drop('target', axis=1)
y = df.target

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# Standard Scaler
sc = StandardScaler()
sc.fit(X_train)

X_train =sc.transform(X_train);
X_test =sc.transform(X_test);

# Logistic Regression
lr = LogisticRegression(C=0.01)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
accuracy_score(y_test, y_pred)

# Save Model
output_file = 'model.pkl'
with open(output_file, 'wb') as f_out:
    pickle.dump((sc, lr), f_out)






