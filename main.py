from clustering446 import *

def evaluate_clustering(dataset, labels):
    """
    Evaluates clustering performance using silhouette score.

    Parameters:
    dataset (ndarray): The dataset, where rows are samples and columns are features.
    labels (ndarray): Cluster labels assigned to each sample.

    Returns:
    float: Silhouette score, or -1 if only one cluster is present.
    """
    # Silhouette score is undefined for single-cluster or unclustered data
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return -1  # Indicates invalid clustering for this metric

    # Compute the silhouette score
    return silhouette_score(dataset, labels)

def test_clustering_algs(dataset, feature_count=2):
    clustering_algorithms = {
        'KMeans': DBSCANClustering('scikit-DBSCAN', dataset, feature_count),
        'DBSCAN': BIRCHClustering('scikit-BIRCH', dataset, feature_count),
        'Birch': kMeansClustering('scikit-KMEANS', dataset, feature_count)
    }
    
    results = {}
    for name, algorithm in clustering_algorithms.items():
        algorithm.cluster()
        algorithm.plot()
        labels = algorithm.clusters
        score = evaluate_clustering(dataset.dataset, labels)
        results[name] = score

    print('\nPERFORMANCE METRICS:\n')
    for method, score in results.items():
        print(f"{method}: Silhouette Score = {score:.3f}")

    plt.show()

def test_2d_moon_dataset():
    # Filling dataset
    moons = Dataset(None)
    moons_dataset = moons.make_moon_dataset(False)
    test_clustering_algs(moons)

def test_2d_coord_dataset():
    # Set an arbitrary limit to minimize computation time (full small dataset has 1048575 rows)
    # DBSCAN does worse the higher this is (for this early implementation)
    limit = 500
    
    with h5py.File('data/twitterSmall.h5', 'r') as file:
        clusters_key = list(file.keys())[0]  # Precomputed clusters ("Clusters")
        coords_key = list(file.keys())[1]  # Coordinates ("DBSCAN")

        coords_dataset = file[coords_key]
        coords_subset = coords_dataset[:limit]
        
        # Displaying some data
        for i in range(limit):
            print(f'Coord #{i}: {coords_subset[i]}')

    coord_object = Dataset(coords_subset)
    test_clustering_algs(coord_object)

if __name__ == '__main__':
    #test_2d_moon_dataset()
    test_2d_coord_dataset()