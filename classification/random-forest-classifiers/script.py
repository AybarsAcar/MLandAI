import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

dataset = pd.read_csv('../../data/Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# create the classifier
# to minimise the information entropy while splitting
classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
y_test_prob = classifier.predict_proba(X_test)
y_pred_vs_test = np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)
print(y_pred_vs_test)

# purchase decision of 30 yr old and estimated salary = 87,000
pred_1 = classifier.predict(scaler.transform([[30, 87000]]))
print(pred_1)

# do it with sklearn accuracy_score function because it loops over the data in c
true_outcome = 0
for i in range(len(y_pred)):
  if y_pred[i] == y_test[i]:
    true_outcome += 1
print('Result:', true_outcome, '/', len(y_pred))

# Confusion Matrix
pred_matrix = confusion_matrix(y_test, y_pred)
print(pred_matrix)
# model struggles making predictions on the purchases, biased towards 1, maybe update the projection threshold
print("Score:", accuracy_score(y_test, y_pred))


# visualise
def visualise(X, y, trainOrTest):
  X_set, y_set = scaler.inverse_transform(X), y
  X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=0.50),
                       np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=0.50))
  plt.figure(figsize=(8, 8))
  plt.contourf(X1, X2, classifier.predict(scaler.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
               alpha=0.75, cmap=ListedColormap(('red', 'green')))
  plt.xlim(X1.min(), X1.max())
  plt.ylim(X2.min(), X2.max())

  for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], color=ListedColormap(('red', 'green'))(i), label=j)

  plt.title(f'KNN Classifier ({trainOrTest} set)')
  plt.xlabel('Age')
  plt.ylabel('Estimated Salary')
  plt.legend()
  plt.show()


# visualisation (train set)
# visualise(X_train, y_train, 'Train')
visualise(X_test, y_test, 'Test')
