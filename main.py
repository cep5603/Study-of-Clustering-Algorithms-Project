from clustering446 import *

def evaluate_clustering(dataset, labels):
    results = {}

    if len(np.unique(labels)) < 2:
        # Consider scores invalid for single cluster
        results['Silhouette Score (higher=better)'] = -1  
        results['Calinski-Harabasz Index (higher=better)'] = -1
        results['Davies-Bouldin Index (lower=better)'] = -1
    else:
        results['Silhouette Score (higher=better)'] = silhouette_score(dataset, labels)
        results['Calinski-Harabasz Index (higher=better)'] = calinski_harabasz_score(dataset, labels)
        results['Davies-Bouldin Index (lower=better)'] = davies_bouldin_score(dataset, labels)

    return results

def test_clustering_algs(dataset, feature_count=2):
    clustering_algorithms = {
        'KMeans': kMeansClustering('scikit-KMEANS', dataset, feature_count),
        'DBSCAN': DBSCANClustering('scikit-DBSCAN', dataset, feature_count, 1),
        'BIRCH': BIRCHClustering('scikit-BIRCH', dataset, feature_count),
        'Spectral': SpectralClustering('scikit-SpectralClustering', dataset, feature_count),
        'HDBSCAN': HDBSCAN('scikit-HDBSCAN', dataset, feature_count)
    }
    
    clustering_algorithms_for_mnist_demo = {
        'KMeans': kMeansClustering('scikit-KMEANS', dataset, feature_count),
        'DBSCAN': DBSCANClustering('scikit-DBSCAN', dataset, feature_count),
        'BIRCH': BIRCHClustering('scikit-BIRCH', dataset, feature_count, 1, 30, 2),
        'HDBSCAN': HDBSCAN('scikit-HDBSCAN', dataset, feature_count)
    }

    clustering_algorithms_for_moons_demo = {
        'KMeans': kMeansClustering('scikit-KMEANS', dataset, feature_count, 2),
        'DBSCAN': DBSCANClustering('scikit-DBSCAN', dataset, feature_count, 1),
        'BIRCH': BIRCHClustering('scikit-BIRCH', dataset, feature_count, 0.2, 8, 2),
        'Spectral': SpectralClustering('scikit-SpectralClustering', dataset, feature_count, 2),
        'HDBSCAN': HDBSCAN('scikit-HDBSCAN', dataset, feature_count)
    }
    
    results = {}
    for name, algorithm in clustering_algorithms_for_mnist_demo.items():
        algorithm.cluster()
        algorithm.plot()
        labels = algorithm.clusters
        print(f'Evaluating {name} . . .')
        scores = evaluate_clustering(dataset.dataset, labels)
        results[name] = scores

    print('====================\nPERFORMANCE METRICS:\n====================')
    metrics = results['KMeans'].keys()  # Just get metrics from KMeans
    for metric in metrics:
        print(f'\nMetric: {metric}')
        for alg, scores in results.items():
            print(f'{alg}: {scores[metric]:.3f}')

    plt.show()

def test_2d_moon_dataset():
    # Filling dataset
    moons = Dataset(None)
    moons_dataset = moons.make_moon_dataset()
    test_clustering_algs(moons)

def test_2d_blobs_dataset():
    blobs = Dataset(None)
    blobs_dataset = blobs.make_blobs_dataset()
    test_clustering_algs(blobs)

def test_2d_coord_dataset(limit):
    # Set an arbitrary limit to minimize computation time (full small dataset has 1,048,575 rows)
    
    with h5py.File('data/twitterSmall.h5', 'r') as file:
        clusters_key = list(file.keys())[0]  # Precomputed clusters ("Clusters")
        coords_key = list(file.keys())[1]  # Coordinates ("DBSCAN")

        coords_dataset = file[coords_key]
        coords_subset = coords_dataset[:limit]

    coord_object = Dataset(coords_subset)
    test_clustering_algs(coord_object)

def test_mnist_dataset(dims_after_pca):
    mnist_values = np.loadtxt("data/tsne_python/mnist2500_X.txt")
    mnist_labels = np.loadtxt("data/tsne_python/mnist2500_labels.txt")
    mnist_dataset = Dataset(None)
    reduced_dataset = mnist_dataset.pca(mnist_values, dims_after_pca).real
    mnist_dataset.dataset = reduced_dataset
    print(f'Finished PCA ({len(mnist_values)} -> {dims_after_pca} dimensions)')
    test_clustering_algs(mnist_dataset)

if __name__ == '__main__':
    #test_2d_blobs_dataset()
    #test_2d_moon_dataset()
    #test_2d_coord_dataset(500)
    test_mnist_dataset(20)