Brady Lauritsen

# How to run it: python kmeans.py <filename> <k> <epsilon> <max_iterations>
    Example: python kmeans.py iris.csv 3 0.001 100
Input meaning: k = clusters, epsilon is the SSD change threshold, and 
max iterations is the safety cutoff.

# Program Design and functionality
- The k-means implementation is designed to handle continuous attributes using a modular, functional approach. 
The program consists of the following key functions:
- read_dataset: Dynamically parses numerical data from a CSV file.
- euclidean_distance: Calculates the distance between data points and centroids.
- initialize_centroids: Randomly selects k initial starting points from the dataset.
- assign_clusters: Assigns each data point to its nearest centroid.
- update_centroids: Recalculates centroids by averaging the coordinates of all points in a cluster.
- compute_ssd: Calculates the Total Sum of Squared Distances to measure clustering goodness.
- kmeans: The core driver that iterates until the SSD change falls below epsilon or the maximum iterations are reached.

# Results
Parameters used: k = 3, epsilon=0.001, max iterations = 100

Observations: The algorithm is sensitive to initial centroid placement. In multiple runs,
the total SSD typically converged around 78.94 though occasionally it settled at a local optimum of 142.86.

Convergence: On average the algorithm converged in 4 to 8 iterations, which satisfies
the condition where the change in SSD fell below epsilon.

- Sample run
Iteration: 0 | SSD: 84.6412
Iteration: 1 | SSD: 83.3704
Iteration: 2 | SSD: 82.0730
Iteration: 3 | SSD: 81.3672
Iteration: 4 | SSD: 80.3157
Iteration: 5 | SSD: 79.6817
Iteration: 6 | SSD: 79.1156
Iteration: 7 | SSD: 78.9451
Iteration: 8 | SSD: 78.9451
Converged based on epsilon.

Final SSD: 78.94506582597728
Final Centroids:
Cluster 0: [5.006, 3.418, 1.464, 0.24400000000000002]
Cluster 1: [5.883606557377049, 2.7409836065573767, 4.388524590163934, 1.4344262295081966]        
Cluster 2: [6.853846153846154, 3.076923076923077, 5.7153846153846155, 2.0538461538461537]