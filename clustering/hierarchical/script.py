import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy as sch

mall_df = pd.read_csv('../../data/Mall_Customers.csv')

X = mall_df.iloc[:, [3, 4]].values


# find the optimal number of clusters using dendogram
def plotdendogram(input_df):
  dendogram = sch.dendrogram(sch.linkage(input_df, method='ward'))
  plt.title('Dendogram')
  plt.xlabel('Customers')
  plt.ylabel('Euclidean distances')
  plt.show()


# plotdendogram(X)

# Instantiate the clustering
cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_predict = cluster.fit_predict(X)
print(y_predict)


def plot_clusters(X, y_predict, n_clusters):
  colours = ['red', 'green', 'blue', 'cyan', 'orange', 'yellow', 'purple', 'pink', 'lime']
  for i in range(0, n_clusters):
    plt.scatter(X[y_predict == i, 0], X[y_predict == i, 1], s=50, c=colours[i], label=f'Cluster {i}')
  plt.title('Cluster of Customers')
  plt.xlabel('Annual Income (k$)')
  plt.ylabel('Spending Score (%)')
  plt.legend()
  plt.show()


plot_clusters(X, y_predict, 5)
