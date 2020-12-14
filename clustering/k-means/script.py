import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

# spending score -> how much the customer spends
mall_df = pd.read_csv('../../data/Mall_Customers.csv')

# print(mall_df.info())
# print(mall_df.head())

X = mall_df.iloc[:, 1:].values

# encode gender
le = LabelEncoder()
X[:, 0] = le.fit_transform(X[:, 0])


# using elbow method to find the optimal number of clusters
def plot_wcss(number_of_clusters):
  wcss = []
  for i in range(1, number_of_clusters + 1):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

  plt.plot(range(1, 11), wcss)
  plt.show()


plot_wcss(10)
# pick 4 as the number of clusters
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=0)
y_kmeans = kmeans.fit_predict(X)

print(y_kmeans)
