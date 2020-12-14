import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Predict the previous Salary of the Candidate
# Positions are directly correlated with he Level, they are encoded to level

# Read data
salary_data = pd.read_csv('../../data/Position_Salaries.csv')
X = salary_data.iloc[:, 1:-1].values
y = salary_data.iloc[:, -1].values

# train the Linear regression model on the whole dataset
linear_regression = LinearRegression()
linear_regression.fit(X, y)

# Train the Polynomial Regression model on the whole dataset
# create the matrix of powered features, preprocess the X features
# degree is that it goes up to X^n
poly_reg = PolynomialFeatures(degree=2)
X_poly_2 = poly_reg.fit_transform(X)  # new matrix of features of X based on position levels

poly_reg_4 = PolynomialFeatures(degree=4)
X_poly_4 = poly_reg_4.fit_transform(X)  # new matrix of features of X based on position levels

linear_regression_poly_2 = LinearRegression()
linear_regression_poly_2.fit(X_poly_2, y)

linear_regression_poly_4 = LinearRegression()
linear_regression_poly_4.fit(X_poly_4, y)

# Visualise the Linear Regression Results
plt.figure(figsize=(8, 8))
plt.scatter(X, y, color='blue')
plt.plot(X, linear_regression.predict(X), color='red')
plt.title('Linear Regression vs Polynomial Regression')
plt.xlabel('Position & Level')
plt.ylabel('Salary')

# Visualising the Polynomial Regression Results with degree n = 2
plt.plot(X, linear_regression_poly_2.predict(X_poly_2), color='yellow')

# Visualising the Polynomial Regression Results with degree n = 4
plt.plot(X, linear_regression_poly_4.predict(X_poly_4), color='green')
plt.show()

# smoother curve plotting
# it is overfitted
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, linear_regression_poly_4.predict(poly_reg_4.transform(X_grid)), color='red')
plt.title('Linear Regression vs Polynomial Regression')
plt.xlabel('Position & Level')
plt.ylabel('Salary')
plt.show()

# Predict a new result with the Linear Regression
# 6.5 is the position level
predict_linear = linear_regression.predict([[6.5]])
print("Linear Prediction", predict_linear)

# Predict a new result with the Polynomial Regression
predict_poly = linear_regression_poly_4.predict(poly_reg_4.fit_transform([[6.5]]))
print("Poly Prediction with n=2", predict_poly)
