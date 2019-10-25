import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

from sklearn.cluster import KMeans

# create dataset
X, y = make_blobs(
   n_samples=100, n_features=2,
   centers=2, cluster_std=0.5,
   shuffle=True, random_state=0
)


#ilk grafik oluşturma
# plot
plt.scatter(
   X[:, 0], X[:, 1],
   c='black', marker='o',
   edgecolor='yellow', s=400  #s: size attributes
)
plt.show()


#kMeans kümeleme algoritması tahmin ayarları

km = KMeans(
    n_clusters=2, init='random',
    n_init=10, max_iter=300, 
    tol=1e-04, random_state=0
)
y_km = km.fit_predict(X)


# plot the 2 clusters
plt.scatter(
    X[y_km == 0, 0], X[y_km == 0, 1],
    s=50, c='green',
    marker='s', edgecolor='black',
    label='Yeşil Küme (Elma)'
)

plt.scatter(
    X[y_km == 1, 0], X[y_km == 1, 1],
    s=50, c='blue',
    marker='o', edgecolor='black',
    label='Mavi Küme(Havuç)'
)

#Son olarak tekrar merkezleme

# plot the centroids
plt.scatter(
    km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
    s=250, marker='*',
    c='red', edgecolor='black',
    label='Küme Merkez'
)
plt.legend(scatterpoints=1)
plt.grid()
plt.show()
