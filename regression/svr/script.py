import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# read the dataset
dataset = pd.read_csv('../../data/Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# no data split
# turn y into a 2 dimensional array int[][]
y = y.reshape(len(y), 1)

# Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()

X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Training the SVR model on the whole dataset
svr = SVR(kernel='rbf')
svr.fit(X, y)

# Predicting the results for someone at level 6.5
prediction = svr.predict(sc_X.transform([[6.5]]))
# Reverse Scaling
prediction = sc_y.inverse_transform(prediction)
print(prediction)

# Make sure to reverse scale the input and result as plotting as well
# Visualising the SVR result
plt.figure(figsize=(8, 8))
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color='red')
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(svr.predict(X)), color='green')
plt.title('Support Vector Regression')
plt.xlabel('Position & Level')
plt.ylabel('Salary')
plt.show()

# smoother plot
X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color='red')
plt.plot(X_grid, sc_y.inverse_transform(svr.predict(sc_X.transform(X_grid))), color='lime')
plt.title('Linear Regression vs Polynomial Regression')
plt.xlabel('Position & Level')
plt.ylabel('Salary')
plt.show()
