import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
from xgboost import XGBClassifier

dataset = pd.read_csv('../../data/Breast_Cancer_Data.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# XGBoost on the Training set
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# scores
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_pred, y_test)
acc = accuracy_score(y_pred, y_test)
print(cm)
print(acc)

# Apply k-Fold Cross Validation
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
print('Accuracy: {:.2f} %'.format(accuracies.mean() * 100))
print('Standard Deviation: {:.2f} %'.format(accuracies.std() * 100))
