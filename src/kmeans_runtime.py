
import csv

data = []

with open('iris.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader, None)  # Skip the header row if it exists
    for row in reader:
        if len(row) >= 4: # Ensure the row has enough columns
            data.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])
        else:
            print(f"Skipping malformed row: {row}")

print("Total data points:", len(data))
if len(data) > 0:
    print("First row:", data[0])
else:
    print("No data points were loaded.")

def euclidean_distance(a, b):
    total = 0
    for i in range(len(a)):
        total += (a[i] - b[i]) ** 2
    return total ** 0.5

import random

def k_means(data, k, epsilon=0.001, max_iterations=100):

    centers = random.sample(data, k)

    for iteration in range(max_iterations):

        clusters = [[] for i in range(k)]

        # assign points to clusters
        for point in data:
            distances = []

            for center in centers:
                distances.append(euclidean_distance(point, center))

            cluster_index = distances.index(min(distances))
            clusters[cluster_index].append(point)

        # compute new centers
        new_centers = []

        for cluster in clusters:
            if len(cluster) == 0:
                new_centers.append(random.choice(data))
            else:
                center = []
                for i in range(len(cluster[0])):
                    center.append(sum(point[i] for point in cluster) / len(cluster))
                new_centers.append(center)

        # check convergence
        change = 0
        for i in range(k):
            change += euclidean_distance(centers[i], new_centers[i]) ** 2

        if change < epsilon:
            break

        centers = new_centers

    return clusters

import time
import matplotlib.pyplot as plt

k_values = list(range(2,11))
runtimes = []

for k in k_values:
    start = time.perf_counter()

    k_means(data, k)

    end = time.perf_counter()

    runtimes.append(end - start)

plt.plot(k_values, runtimes, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Runtime (seconds)")
plt.title("Runtime vs Number of Clusters")
plt.show()

dimensions = [1,2,3,4]
runtimes = []

for d in dimensions:

    subset = [row[:d] for row in data]

    start = time.perf_counter()

    k_means(subset, 3)

    end = time.perf_counter()

    runtimes.append(end - start)

plt.plot(dimensions, runtimes, marker='o')
plt.xlabel("Number of Dimensions")
plt.ylabel("Runtime (seconds)")
plt.title("Runtime vs Number of Dimensions")
plt.show()

sizes = [30,60,90,120,150]
runtimes = []

for size in sizes:

    subset = data[:size]

    start = time.perf_counter()

    k_means(subset, 3)

    end = time.perf_counter()

    runtimes.append(end - start)

plt.plot(sizes, runtimes, marker='o')
plt.xlabel("Dataset Size")
plt.ylabel("Runtime (seconds)")
plt.title("Runtime vs Dataset Size")
plt.show()