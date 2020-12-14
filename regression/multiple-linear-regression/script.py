import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# read the data as DataFrame
startup_data = pd.read_csv("../../data/50_Startups.csv")

# print(startup_data.info())

X = startup_data.iloc[:, :-1].values
y = startup_data.iloc[:, -1].values

# One-hot encoding column 3, hot encoded columns are placed in the beginning
# 3 categories, 3 columns 0,1,2 are the encoded columns
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the multiple linear regression model on the training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)  # 2 decimal points when printing np arrays

# concatenate the real profits and the predicted profit vertically
concatted = np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_pred), 1)), axis=1)
# print(concatted)

# Making a single prediction for example:
# the profit of a startup with R&D Spend = 160000,
# Administration Spend = 130000,
# Marketing Spend = 300000,
# State = 'California'
prediction = regressor.predict([[1, 0, 0, 160000, 130000, 300000]])
print(prediction)