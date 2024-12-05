from abc import ABC, abstractmethod
import h5py  # https://docs.h5py.org/en/stable/index.html
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kneed  # https://kneed.readthedocs.io/en/stable/
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
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
    
    def get_k_value(self, max_k=10):
        """
        Determines the ideal number of clusters (k) using the elbow method.

        Parameters:
        max_k (int): Maximum number of clusters to test. Default is 10.

        Returns:
        int: The ideal number of clusters.
        """
        distortions = []
        k_range = range(1, max_k + 1)

        # Compute distortions for each k
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(self.dataset)
            distortions.append(kmeans.inertia_)

        knee_locator = kneed.KneeLocator(k_range, distortions, curve='convex', direction='decreasing')
        return knee_locator.knee or 1  # Default to 1 cluster if no knee is found

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
        k = self.dataset_2d.get_k_value()
        kmeans = KMeans(n_clusters=k, random_state=self.seed)  # k-Means also uses a (separate) seed
        self.clusters = kmeans.fit_predict(self.dataset_2d.dataset)

class DBSCANClustering(ClusteringAlgorithm):
    def __init__(self, name, dataset_2d, dim_count):
        super().__init__(name, dataset_2d, dim_count)
        self.epsilon = self.dataset_2d.get_epsilon()
        self.min_pts = self.dim_count * 2 + 1

    def cluster(self):
        dbscan = DBSCAN(eps=self.epsilon, min_samples=self.min_pts)
        self.clusters = dbscan.fit_predict(self.dataset_2d.dataset)

class BIRCHClustering(ClusteringAlgorithm):
    # TODO: Params in init()
    def cluster(self):
        # TODO: Other values for threshold/bf
        thresholds = np.linspace(0.1, 1.0, 10)
        branching_factors = range(10, 101, 10)

        results = self.optimize_birch(thresholds, branching_factors)
        print("Best Score:", results['best_score'])
        print("Best Parameters:", results['best_params'])

        birch = Birch(
            threshold=results['best_params']['threshold'],
            branching_factor=results['best_params']['branching_factor']
        )

        self.clusters = birch.fit_predict(self.dataset_2d.dataset)

    def optimize_birch(self, thresholds, branching_factors):
        """
        Optimizes the Birch clustering algorithm parameters: threshold and branching_factor.

        Parameters:
        thresholds (list): List of threshold values to test.
        branching_factors (list): List of branching factor values to test.

        Returns:
        dict: Best parameters and corresponding silhouette score.
        """
        best_score = -1
        best_params = {'threshold': None, 'branching_factor': None}

        for threshold in thresholds:
            for branching_factor in branching_factors:
                # Initialize BIRCH
                birch = Birch(threshold=threshold, branching_factor=branching_factor)
                clusters = birch.fit_predict(self.dataset_2d.dataset)

                # Skip if fewer than 2 clusters (silhouette score is undefined)
                if len(np.unique(clusters)) > 1:
                    score = silhouette_score(self.dataset_2d.dataset, clusters)
                    if score > best_score:
                        best_score = score
                        best_params['threshold'] = threshold
                        best_params['branching_factor'] = branching_factor

        return {'best_score': best_score, 'best_params': best_params}