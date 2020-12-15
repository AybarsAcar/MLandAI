import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

dataset = pd.read_csv('../../data/Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Training Kernel SVM model
classifier = SVC(kernel='rbf', random_state=0)
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

# Apply Grid Search
parameters = [
  {
    'C': [0.25, 0.5, 0.75, 1],
    'kernel': ['linear']
  },
  {
    'C': [0.25, 0.5, 0.75, 1],
    'kernel': ['rbf'],
    'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
  }
]

grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10, n_jobs=-1)

# connect the object on the training set
grid_search.fit(X_train, y_train)

best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print('Best Accuracy: {:.2f} %'.format(best_accuracy * 100))
print('Standard Parameters:', best_parameters)

