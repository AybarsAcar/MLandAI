import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# read in the data as a DataFrame
data_set = pd.read_csv("../data/Data.csv")

# preprocess data
# X -> all the rows and first 3 columns except the last one
X = data_set.iloc[:, :-1].values
# y -> dependent variable, all the rows and last column
y = data_set.iloc[:, -1].values

# processing the missing variables by their mean
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# make sure to exclude the String[] column
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# one-hot encoding the country column which is at index = 0
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
# make sure to cast is as a numpy array which our ML model is expecting
X = np.array(ct.fit_transform(X))

# Encode the dependent variable which is a binary output Yes and No
le = LabelEncoder()
y = le.fit_transform(y)  # this doesnt have to be a numpy array

# Split the Data into Train and Test, remove random_state after model development
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Feature Scaling
sc = StandardScaler()

# exclude the first 3 columns because they represent the country
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
