import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

dataset = pd.read_csv('../../data/Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# instantiate the regressor, random state to 0 to get consistent test result
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

# prediction for the position level 6.5
prediction = regressor.predict([[6.5]])
# not a good prediction because not too many features in the data
print(prediction)

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.plot('Decision Tree Model')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()