from abc import ABC, abstractmethod
import h5py  # https://docs.h5py.org/en/stable/index.html
import numpy as np
import matplotlib.pyplot as plt
import kneed  # https://kneed.readthedocs.io/en/stable/
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import make_moons

# Class definitions

class Dataset:
    def __init__(self, dataset):
        self.seed = 100
        self.k = 5
        self.dataset = dataset

    def pca(self):
        # TODO: Feature reduction on high-dimensional data (see t-SNE)
        pass
    
    def get_k_nearest_neighbors_distances(self):
        neigh = NearestNeighbors(n_neighbors=self.k)
        neigh.fit(self.dataset)
        distances, _ = neigh.kneighbors(self.dataset)
        distances = np.sort(distances[:, self.k - 1])
        return distances

    def get_epsilon(self):
        # Use kneed to get "good" epsilon value from elbow in k-distances graph
        distances = self.get_k_nearest_neighbors_distances()
        kneedle = kneed.KneeLocator(range(len(distances)), distances, curve='convex', direction='increasing')
        print(f'Epsilon value found: {distances[kneedle.knee]}\n')
        return distances[kneedle.knee]

    def plot_k_distance_graph(self):
        # Visually plot k-distance graph, if desired
        plt.figure(figsize=(10, 6))
        plt.plot(self.get_k_nearest_neighbors_distances())
        plt.xlabel('Points')
        plt.ylabel(f'{self.k}-th nearest neighbor distance')
        plt.title('K-distance Graph')
        plt.show()

    def make_moon_dataset(self, plot_k_graph = False):
        self.dataset, _ = make_moons(n_samples=200, noise=0.05, random_state=self.seed)

        if (plot_k_graph):
            self.plot_k_distance_graph()

        # Assign to self and return data for now
        return self.dataset

class ClusteringAlgorithm(ABC):
    def __init__(self, name, dataset_2d, dim_count):
        self.seed = 100
        self.name = name
        #self.params = params  # TODO
        self.dataset_2d = dataset_2d
        self.dim_count = dim_count

    @abstractmethod
    def cluster(self):
        pass

    def plot(self):
        # Visualize the clusters
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(self.dataset_2d.dataset[:, 0], self.dataset_2d.dataset[:, 1], c=self.clusters, cmap='viridis')
        plt.colorbar(scatter)
        plt.title(f'{self.name} Clustering Results')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')

class kMeansClustering(ClusteringAlgorithm):
    def cluster(self):
        kmeans = KMeans(n_clusters=self.dim_count, random_state=self.seed)  # k-Means also uses a (separate) seed
        self.clusters = kmeans.fit_predict(self.dataset_2d.dataset)

class DBSCANClustering(ClusteringAlgorithm):
    def __init__(self, name, dataset_2d, dim_count):
        super().__init__(name, dataset_2d, dim_count)
        self.epsilon = self.dataset_2d.get_epsilon()
        self.min_pts = self.dim_count * 2 + 1

    def cluster(self):
        dbscan = DBSCAN(eps=self.epsilon, min_samples=self.min_pts)
        self.clusters = dbscan.fit_predict(self.dataset_2d.dataset)