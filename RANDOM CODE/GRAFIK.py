import pandas as pd

# Data asli
data = pd.DataFrame({
    'Titik': ['X1', 'X2', 'X3', 'X4'],
    'x': [1, 0, 2, 3],
    'y': [0, 1, 1, 3]
})
# Inisialisasi cluster acak (Iterasi 0)
# Inisialisasi cluster acak (Iterasi 0)
data['Cluster_0'] = ['C1', 'C2', 'C1', 'C2']

def compute_centroid(df, col_cluster):
    centroids = {}
    for c in df[col_cluster].unique():
        subset = df[df[col_cluster] == c]
        centroids[c] = (subset['x'].mean(), subset['y'].mean())
    return centroids
#mantap
# Iterasi 1
centroids_0 = compute_centroid(data, 'Cluster_0')
for c, (cx, cy) in centroids_0.items():
    data[f'dsq_{c}_0'] = (data['x'] - cx)**2 + (data['y'] - cy)**2
data['Cluster_1'] = data.apply(lambda row: 'C1' if row['dsq_C1_0'] < row['dsq_C2_0'] else 'C2', axis=1)

# Iterasi 2
centroids_1 = compute_centroid(data, 'Cluster_1')
for c, (cx, cy) in centroids_1.items():
    data[f'dsq_{c}_1'] = (data['x'] - cx)**2 + (data['y'] - cy)**2
data['Cluster_2'] = data.apply(lambda row: 'C1' if row['dsq_C1_1'] < row['dsq_C2_1'] else 'C2', axis=1)

# Tampilkan
print("Data Asli dengan Cluster Inisialisasi:")
print(data[['Titik','x','y','Cluster_0']])
print("\nHasil Iterasi 1:")
print(data[['Titik','x','y','Cluster_1','dsq_C1_0','dsq_C2_0']])
print("\nHasil Iterasi 2:")
print(data[['Titik','x','y','Cluster_2','dsq_C1_1','dsq_C2_1']])
