from abc import ABC, abstractmethod
import h5py  # https://docs.h5py.org/en/stable/index.html
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kneed  # https://kneed.readthedocs.io/en/stable/
import sklearn.cluster
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.datasets import make_moons, make_blobs

import os

os.environ['OMP_NUM_THREADS'] = '1'

# Class definitions

class Dataset:
    def __init__(self, dataset):
        self.seed = 100
        self.k = 5
        self.dataset = dataset

    def pca(self, X=np.array([]), no_dims=50):
        """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
        """

        (n, d) = X.shape
        X = X - np.tile(np.mean(X, 0), (n, 1))
        (l, M) = np.linalg.eig(np.dot(X.T, X))
        Y = np.dot(X, M[:, 0:no_dims])
        return Y
    
    def get_k_nearest_neighbors_distances(self):
        neigh = NearestNeighbors(n_neighbors=self.k)
        neigh.fit(self.dataset)
        distances, _ = neigh.kneighbors(self.dataset)
        distances = np.sort(distances[:, self.k - 1])
        return distances

    def get_epsilon(self, sensitivity, plot_k_graph = False):
        # Use kneed to get "good" epsilon value from elbow in k-distances graph
        distances = self.get_k_nearest_neighbors_distances()
        kneedle = kneed.KneeLocator(range(len(distances)), distances, curve='convex', direction='increasing', S=sensitivity)
        print(f'(H)DBSCAN epsilon value: {kneedle.elbow_y}\n')
        if (plot_k_graph):
            self.plot_k_distance_graph()
            kneedle.plot_knee_normalized()
        return distances[kneedle.knee]

    def plot_k_distance_graph(self):
        # Visually plot k-distance graph, if desired
        distances = self.get_k_nearest_neighbors_distances()
        plt.figure(figsize=(10, 6))
        plt.plot(distances)
        plt.xlabel('Points')
        plt.ylabel(f'{self.k}-th nearest neighbor distance')
        plt.title('K-distance Graph')
        #plt.show()

    def make_moon_dataset(self):
        self.dataset, _ = make_moons(n_samples=200, noise=0.05, random_state=self.seed)
        return self.dataset
    
    def make_blobs_dataset(self):
        self.dataset, _ = make_blobs(n_samples=100, centers=3, n_features=2,random_state=self.seed)
        return self.dataset

class ClusteringAlgorithm(ABC):
    def __init__(self, name, stored_dataset, dim_count):
        self.seed = 100
        self.name = name
        self.stored_dataset = stored_dataset
        self.dim_count = dim_count

    @abstractmethod
    def cluster(self):
        pass

    def plot(self):
        # Visualize the clusters
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(self.stored_dataset.dataset[:, 0], self.stored_dataset.dataset[:, 1], c=self.clusters, cmap='viridis')
        plt.colorbar(scatter)
        plt.title(f'{self.name} Clustering Results')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')

class kMeansClustering(ClusteringAlgorithm):
    def __init__(self, name, stored_dataset, dim_count, num_clusters = None):
        super().__init__(name, stored_dataset, dim_count)
        self.num_clusters = num_clusters
    def cluster(self):
        if self.num_clusters is None:
            self.num_clusters = self.get_k_value(self.stored_dataset.dataset)
        
        print('Clustering KMeans . . .')
        kmeans = sklearn.cluster.KMeans(n_clusters=self.num_clusters, random_state=self.seed)
        self.clusters = kmeans.fit_predict(self.stored_dataset.dataset)

    def get_k_value(self, dataset, max_k=10):
        """
        Determines the ideal number of clusters (k) using the elbow method.

        Parameters:
        max_k (int): Maximum number of clusters to test. Default is 10.

        Returns:
        int: The ideal number of clusters.
        """
        print('Optimizing KMeans . . .')
        distortions = []
        k_range = range(1, max_k + 1)

        # Compute distortions for each k
        for k in k_range:
            kmeans = sklearn.cluster.KMeans(n_clusters=k, random_state=self.seed)
            kmeans.fit(dataset)
            distortions.append(kmeans.inertia_)

        knee_locator = kneed.KneeLocator(k_range, distortions, curve='convex', direction='decreasing')
        return knee_locator.knee or 1  # Default to 1 cluster if no knee is found

class DBSCANClustering(ClusteringAlgorithm):
    def __init__(self, name, stored_dataset, dim_count, sensitivity = 2):
        super().__init__(name, stored_dataset, dim_count)
        # For now, can adjust sensitivity here
        self.epsilon = self.stored_dataset.get_epsilon(sensitivity)
        self.min_pts = self.dim_count * 2 + 1

    def cluster(self):
        print('Clustering DBSCAN . . .')
        dbscan = sklearn.cluster.DBSCAN(eps=self.epsilon, min_samples=self.min_pts)
        self.clusters = dbscan.fit_predict(self.stored_dataset.dataset)

class BIRCHClustering(ClusteringAlgorithm):
    def __init__(self, name, stored_dataset, dim_count, threshold=None, branching_factor=None, num_clusters=None):
        super().__init__(name, stored_dataset, dim_count)
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.num_clusters = num_clusters

    def cluster(self):
        # Just check for all 3 for simplicity
        if self.threshold is not None and self.branching_factor is not None and self.num_clusters is not None:
            birch = sklearn.cluster.Birch(
                threshold=self.threshold,
                branching_factor=self.branching_factor,
                n_clusters=self.num_clusters
            )
        else:
            # Possible values here are done to limit computation; better values may be outside of these ranges
            thresholds = np.linspace(0.1, 1.0, 10)
            branching_factors = range(10, 101, 10)
            num_clusters = range(2, 10)

            results = self.optimize_birch(thresholds, branching_factors, num_clusters)
            print("BIRCH Best Score:", results['best_score'])
            print("BIRCH Best Parameters:", results['best_params'])
            print()

            birch = sklearn.cluster.Birch(
                threshold=results['best_params']['threshold'],
                branching_factor=results['best_params']['branching_factor'],
                n_clusters=results['best_params']['num_clusters']
            )

        print('Clustering BIRCH . . .')
        self.clusters = birch.fit_predict(self.stored_dataset.dataset)

    def optimize_birch(self, thresholds, branching_factors, num_clusters):
        """
        Optimizes the Birch clustering algorithm parameters: threshold and branching_factor.

        Parameters:
        thresholds (list): List of threshold values to test.
        branching_factors (list): List of branching factor values to test.

        Returns:
        dict: Best parameters and corresponding silhouette score.
        """
        print('Optimizing parameters for BIRCH (this can take some time!)')

        best_score = -1
        best_params = {'threshold': None, 'branching_factor': None, 'num_clusters': None}

        for threshold in thresholds:
            for branching_factor in branching_factors:
                for num_cluster in num_clusters:
                    # Initialize BIRCH
                    birch = sklearn.cluster.Birch(threshold=threshold, branching_factor=branching_factor, n_clusters=num_cluster)
                    clusters = birch.fit_predict(self.stored_dataset.dataset)

                    # Skip if fewer than 2 clusters (silhouette score is undefined)
                    if len(np.unique(clusters)) > 1:
                        score = silhouette_score(self.stored_dataset.dataset, clusters)
                        if score > best_score:
                            best_score = score
                            best_params['threshold'] = threshold
                            best_params['branching_factor'] = branching_factor
                            best_params['num_clusters'] = num_cluster

        return {'best_score': best_score, 'best_params': best_params}

class SpectralClustering(ClusteringAlgorithm):
    def __init__(self, name, stored_dataset, dim_count, num_clusters=None):
        super().__init__(name, stored_dataset, dim_count)
        self.num_clusters = num_clusters

    def cluster(self):
        if self.num_clusters is None:
            self.num_clusters = self.optimize_spectral()
        
        print('Clustering Spectral . . .')
        spectral = sklearn.cluster.SpectralClustering(n_clusters=self.num_clusters, random_state=self.seed)
        self.clusters = spectral.fit_predict(self.stored_dataset.dataset)

    def optimize_spectral(self):
        print('Optimizing parameters for Spectral Clustering')
        num_clusters = range(2, 20)
        best_cluster_count = None
        best_score = -1

        for num_cluster in num_clusters:
            spectral = sklearn.cluster.SpectralClustering(n_clusters=num_cluster, random_state=self.seed)
            clusters = spectral.fit_predict(self.stored_dataset.dataset)

            # Skip if fewer than 2 clusters (silhouette score is undefined)
            if len(np.unique(clusters)) > 1:
                score = silhouette_score(self.stored_dataset.dataset, clusters)
                #print(f'Cluster count {num_cluster} has score {score}')
                if score > best_score:
                    best_score = score
                    best_cluster_count = num_cluster

        print(f'Optimal number of clusters for spectral: {best_cluster_count}')
        return best_cluster_count

class HDBSCAN(ClusteringAlgorithm):
    def __init__(self, name, stored_dataset, dim_count):
        super().__init__(name, stored_dataset, dim_count)

    def cluster(self):
        print('Clustering HDBSCAN . . .')
        hdbscan = sklearn.cluster.HDBSCAN()
        self.clusters = hdbscan.fit_predict(self.stored_dataset.dataset)