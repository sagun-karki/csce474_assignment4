import sys
import random
import math
import csv
import time
import matplotlib.pyplot as plt

def read_dataset(filename):
    raw_data = []
    with open(filename, "r") as f:
        for line in f:
            if not line.strip(): continue 
            values = line.strip().split(",")
            try:
                # Dynamically read all columns except the last one (usually the label) [cite: 49]
                row = [float(x) for x in values[:-1]] 
                raw_data.append(row)
            except ValueError:
                continue
    
    if not raw_data:
        return []

    # --- Min-Max Normalization Logic ---
    num_vars = len(raw_data[0])
    num_rows = len(raw_data)
    
    # Find min and max for each column
    col_min = [min(row[i] for row in raw_data) for i in range(num_vars)]
    col_max = [max(row[i] for row in raw_data) for i in range(num_vars)]
    
    normalized_data = []
    for row in raw_data:
        norm_row = []
        for i in range(num_vars):
            denominator = col_max[i] - col_min[i]
            # Prevent division by zero if all values in a column are the same
            if denominator == 0:
                norm_row.append(0.0)
            else:
                # Apply (x - min) / (max - min) 
                norm_row.append((row[i] - col_min[i]) / denominator)
        normalized_data.append(norm_row)
        
    return normalized_data

def euclidean_distance(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

def compute_ssd(clusters, centroids):
    ssd = 0.0
    for i, cluster in enumerate(clusters):
        for point in cluster:
            ssd += euclidean_distance(point, centroids[i]) ** 2
    return ssd

def kmeans(data, k, epsilon, max_iterations):
    if k > len(data):
        raise ValueError("k cannot be greater than dataset size")
    
    centroids = random.sample(data, k)
    prev_ssd = float('inf')
    
    for _ in range(max_iterations):
        clusters = [[] for _ in range(k)]
        
        # Assignment
        for point in data:
            dists = [euclidean_distance(point, c) for c in centroids]
            clusters[dists.index(min(dists))].append(point)
        
        # Update
        new_centroids = []
        for cluster in clusters:
            if not cluster:
                new_centroids.append(random.choice(data))
            else:
                dim = len(cluster[0])
                center = [sum(p[i] for p in cluster) / len(cluster) for i in range(dim)]
                new_centroids.append(center)
        
        # Convergence Check 
        current_ssd = compute_ssd(clusters, new_centroids)
        if abs(prev_ssd - current_ssd) < epsilon:
            break
            
        centroids = new_centroids
        prev_ssd = current_ssd
        
    return clusters, centroids, current_ssd

def plot_runtime_vs_k(data, epsilon, max_iter):
    k_range = list(range(2, 11))
    times = []
    for k in k_range:
        start = time.perf_counter()
        kmeans(data, k, epsilon, max_iter)
        times.append(time.perf_counter() - start)
    
    plt.figure()
    plt.plot(k_range, times, marker='o')
    plt.title("Runtime vs Number of Clusters")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Runtime (seconds)")
    plt.savefig("runtime_vs_k.png")
    plt.close()

def plot_runtime_vs_dims(data, k, epsilon, max_iter):
    max_dim = len(data[0])
    dims = list(range(1, max_dim + 1))
    times = []
    for d in dims:
        subset = [row[:d] for row in data]
        start = time.perf_counter()
        kmeans(subset, k, epsilon, max_iter)
        times.append(time.perf_counter() - start)
    
    plt.figure()
    plt.plot(dims, times, marker='o')
    plt.title("Runtime vs Number of Dimensions")
    plt.xlabel("Number of Dimensions")
    plt.ylabel("Runtime (seconds)")
    plt.savefig("runtime_vs_dims.png")
    plt.close()

def plot_runtime_vs_size(data, k, epsilon, max_iter):
    sizes = [int(len(data) * i) for i in [0.2, 0.4, 0.6, 0.8, 1.0]]
    sizes = [max(10, s) for s in sizes] # Ensure min size
    times = []
    for s in sizes:
        subset = data[:s]
        start = time.perf_counter()
        kmeans(subset, k, epsilon, max_iter)
        times.append(time.perf_counter() - start)
    
    plt.figure()
    plt.plot(sizes, times, marker='o')
    plt.title("Runtime vs Dataset Size")
    plt.xlabel("Dataset Size")
    plt.ylabel("Runtime (seconds)")
    plt.savefig("runtime_vs_size.png")
    plt.close()

def plot_goodness(data, k_max, epsilon, max_iter):
    k_range = list(range(1, k_max + 1))
    ssds = []
    for k in k_range:
        _, _, ssd = kmeans(data, k, epsilon, max_iter)
        ssds.append(ssd)
    
    plt.figure()
    plt.plot(k_range, ssds, marker='o', linestyle='-')
    plt.title("Goodness of Clustering (Elbow Method)")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Total SSD")
    plt.grid(True)
    plt.savefig("goodness_of_clustering.png")
    plt.close()
    print(f"Optimal k estimation saved to goodness_of_clustering.png")

def main():
    if len(sys.argv) != 5:
        print("Usage: python kmeans_final.py <data.csv> <k> <epsilon> <max_iterations>")
        sys.exit(1)

    filename = sys.argv[1]
    k = int(sys.argv[2])
    epsilon = float(sys.argv[3])
    max_iter = int(sys.argv[4])

    data = read_dataset(filename)
    if not data:
        print("Error: No valid data loaded.")
        sys.exit(1)

    print(f"Running K-Means with k={k}, epsilon={epsilon}, max_iter={max_iter}...")
    
    # 1. Run specific instance
    clusters, centroids, final_ssd = kmeans(data, k, epsilon, max_iter)
    print(f"Final SSD: {final_ssd:.4f}")
    print(f"Centroids: {centroids}")

    # 2. Generate Required Plots
    print("Generating runtime plots...")
    plot_runtime_vs_k(data, epsilon, max_iter)
    plot_runtime_vs_dims(data, k, epsilon, max_iter)
    plot_runtime_vs_size(data, k, epsilon, max_iter)
    
    print("Generating goodness plot...")
    plot_goodness(data, 10, epsilon, max_iter)

    print("Done. Check generated .png files.")

if __name__ == "__main__":
    main()