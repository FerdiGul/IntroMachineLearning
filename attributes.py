from sklearn.cluster import KMeans
import numpy as np

X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0]])
              
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

print("\nLabels_: {} \n".format(kmeans.labels_))
print("\nPredict: {} \n".format(kmeans.predict([[0, 0], [12, 3]])))
print("\ncluster_centers_: {} \n".format(kmeans.cluster_centers_))
