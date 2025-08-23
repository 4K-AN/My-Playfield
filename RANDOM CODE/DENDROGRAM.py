import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import numpy as np

# Linkage matrix: [idx1, idx2, distance, cluster_size]
linkage = np.array([
    [0, 3, 1.414, 2],   # Merge point 1 (index 0) & 4 (index 3)
    [4, 1, 1.732, 3],   # Merge cluster(0,3) with point 2 (index 1)
    [5, 2, 3.000, 4]    # Merge cluster above with point 3 (index 2)
])
#fwgwgsgwg
# Plot dendrogram
dendrogram(linkage, labels=["1", "2", "3", "4"])
plt.title("Dendrogram - Single Linkage")
plt.xlabel("Data Point")
plt.ylabel("Jarak (Distance)")
plt.show()
