from clustering446 import *

def test_clustering_algs(dataset, feature_count=2):
    mydbscan = DBSCANClustering('scikit-DBSCAN', dataset, feature_count)
    mydbscan.cluster()
    mydbscan.plot()

    mybirch = BIRCHClustering('scikit-BIRCH', dataset, feature_count)
    mybirch.cluster()
    mybirch.plot()

    # TODO: assign good k value
    mykmeans = kMeansClustering('scikit-KMEANS', dataset, feature_count)
    mykmeans.cluster()
    mykmeans.plot()

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