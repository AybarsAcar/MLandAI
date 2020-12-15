import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def plot(X, y, title):
  plt.figure(figsize=(8, 8))
  plt.scatter(X, y, color='blue')
  plt.plot(X_train, regressor.predict(X_train), color='red')
  plt.title(title)
  plt.xlabel('Years of Experience')
  plt.ylabel('Salary')
  plt.show()


# read the data
dataset = pd.read_csv('../../data/Salary_Data.csv')

# Split the data into training and testing
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

regressor = LinearRegression()

# train the model
regressor.fit(X_train, y_train)

# predict the result
y_pred = regressor.predict(X_test)

# visualise the training set result
# plot(X_train, y_train, "Salary vs Experience (Training Set)")

# visualise the test set result
# plot(X_test, y_test, "Salary vs Experience (Test Set)")

# predict for employee with 12 years of experience
# predict model arguement expects a matrix <T>[][]
print(regressor.predict([[12]]))

import pickle
with open('regression_model', 'wb') as f:
  pickle.dump(regressor, f)

