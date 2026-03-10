import sys
import random
import math

def read_dataset(filename):
    data = []
    with open(filename, "r") as f:
        for line in f:
            if not line.strip(): continue # Skip empty lines
            values = line.strip().split(",")
            try:
                # Dynamically read all columns except the last one (usually the label)
                # If your CSV only has data, change this to values[:]
                row = [float(x) for x in values[:-1]] 
                data.append(row)
            except ValueError:
                continue
    return data

def euclidean_distance(p1, p2):
    total = sum((p1[i] - p2[i]) ** 2 for i in range(len(p1)))
    return math.sqrt(total)

def initialize_centroids(data, k):
    return random.sample(data, k)

def assign_clusters(data, centroids):
    clusters = [[] for _ in range(len(centroids))]
    for point in data:
        distances = [euclidean_distance(point, c) for c in centroids]
        closest = distances.index(min(distances))
        clusters[closest].append(point)
    return clusters

def update_centroids(clusters):
    new_centroids = []
    for cluster in clusters:
        if not cluster: # Handle potential empty clusters
            continue
        dims = len(cluster[0])
        centroid = [sum(p[i] for p in cluster) / len(cluster) for i in range(dims)]
        new_centroids.append(centroid)
    return new_centroids

def compute_ssd(clusters, centroids):
    ssd = 0
    for i, cluster in enumerate(clusters):
        for point in cluster:
            ssd += euclidean_distance(point, centroids[i]) ** 2
    return ssd

def kmeans(data, k, epsilon, max_iterations):
    centroids = initialize_centroids(data, k)
    previous_ssd = float("inf")

    for iteration in range(max_iterations):
        clusters = assign_clusters(data, centroids)
        new_centroids = update_centroids(clusters)
        
        # In k-means, if a cluster becomes empty, the number of centroids might change.
        # We update our reference centroids here.
        centroids = new_centroids
        ssd = compute_ssd(clusters, centroids)

        print(f"Iteration: {iteration} | SSD: {ssd:.4f}")

        # Check for convergence based on change in SSD [cite: 10, 12]
        if abs(previous_ssd - ssd) < epsilon:
            print("Converged based on epsilon.")
            break
        previous_ssd = ssd
    else:
        print("Reached maximum iterations.")

    return clusters, centroids

def main():
    if len(sys.argv) < 5:
        print("Usage: python kmeans.py <file> <k> <epsilon> <max_iterations>")
        sys.exit(1)

    filename = sys.argv[1]
    k = int(sys.argv[2])
    epsilon = float(sys.argv[3])
    max_iterations = int(sys.argv[4])

    data = read_dataset(filename)
    clusters, centroids = kmeans(data, k, epsilon, max_iterations)

    print("\nFinal SSD:", compute_ssd(clusters, centroids))
    print("Final Centroids:")
    for i, c in enumerate(centroids):
        print(f"Cluster {i}: {c}")

if __name__ == "__main__":
    main()