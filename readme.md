Abstract
Clustering is a major research area in data science, enabling data grouping based on
similarity metrics to enhance human interpretability. Given the self-augmenting knowledge
demands of organizations in e-commerce, finance, healthcare, logistics, and other fields, the
application of clustering algorithms to high-dimensional datasets is a subject with particularly
notable research and engineering attention. Many clustering algorithms are in use today—with
considerable differences in parameters, metrics, scalability, and geometry—so some, of course,
are better suited to certain types of data than others. It is sometimes a challenge, then, to
determine which method of clustering is best for a given need. This project presents a
comparative study of the effectiveness of clustering algorithms—including DBSCAN,
HDBSCAN, BIRCH, spectral, and k-means clustering—on both low- (spatial coordinates) and
high- (MNIST text data) dimensional datasets. To reduce the dimensionality of the data, the t-
distributed stochastic neighbor embedding (t-SNE) algorithm will be used, though this project
may also compare other algorithms. The performance of these algorithms will be evaluated on
the bases of clustering quality, scalability and efficiency, and interpretability, using methods like
Silhouette Score, Adjusted Rand Index (ARI), and Normalized Mutual Information (NMI) as
numeric quantifiers.
Goal
To evaluate the performance of commonly used clustering algorithms on datasets with
distinct dimensional structures to understand their applicability, limitations, and clustering
efficacy in high- and low-dimensional data contexts.

Objectives
Data pre-processing:
• The two starting datasets chosen are already cleaned and normalized, so there will be no
need to map the text data onto word embedding vectors. However, this project will still
perform PCA on the text data to vastly reduce its dimensions, with tests conducted to see
what numbers of dimensions offer better clustering performance for different algorithms.
• Reading of datasets from local files:
o Python libraries used: HDF5, NumPy, pandas
Algorithm implementation:
• Implementation + application of clustering algorithms:
o Python libraries used: scikit-learn, matplotlib, NumPy, pandas
Visualization and interpretation:
• Representation of clustering on 1D (low-dimension) or 2D (high-dimension) projections,
using t-SNE:
o Python libraries used: matplotlib, NumPy
Performance evaluation:
• This project will evaluate clustering results with numerical metrics (e.g. Silhouette Score,
ARI, NMI).
• After this, we will analyze how dimensionality and dataset structure impact clustering
effectiveness and accuracy for each algorithm.


# References

[K-DBSCAN: An improved DBSCAN algorithm for big data](https://link.springer.com/article/10.1007/s11227-020-03524-3)

[Genetic Algorithm-Based Optimization of Clustering Algorithms for the Healthy Aging Dataset](https://www.mdpi.com/2076-3417/14/13/5530)

[A Rapid Review of Clustering Algorithms](https://ar5iv.labs.arxiv.org/html/2401.07389)

[Interpretable Clustering: A Survey](https://ar5iv.labs.arxiv.org/html/2409.00743)

[scikit-learn - 2.3.1. Overview of clustering methods](https://scikit-learn.org/stable/modules/clustering.html#overview-of-clustering-methods)

[A Friendly Introduction to Text Clustering](https://towardsdatascience.com/a-friendly-introduction-to-text-clustering-fa996bcefd04)

[Advances in Clustering Algorithms for Large-Scale Data Processing with AI](https://ieeexplore.ieee.org/document/10441990)

[t-SNE](https://lvdmaaten.github.io/tsne/)

[Supplemental Material for Visualizing Data using t-SNE](https://lvdmaaten.github.io/publications/misc/Supplement_JMLR_2008.pdf)

[Probabilistic Topic Models](https://www.cs.columbia.edu/~blei/papers/Blei2012.pdf)