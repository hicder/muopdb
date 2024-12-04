import h5py
import numpy as np

# This script creates a random dataset with 10 clusters, each with 1000 points,
# and 128 dimensions. The data is saved to a HDF5 file.

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
n_clusters = 10
points_per_cluster = 1000
n_dimensions = 128  # 128-dimensional data
std_dev = 5.0  # Standard deviation for the normal distribution

# Initialize array to store all points
all_points = []

# Generate points for each cluster
for i in range(n_clusters):
    # Create cluster center: [i*100, i*100, i*100, ...]
    center = np.array([i * 100.0] * n_dimensions)
    
    # Generate points around the center using normal distribution
    cluster_points = np.random.normal(
        loc=center,
        scale=std_dev,
        size=(points_per_cluster, n_dimensions)
    )
    
    all_points.append(cluster_points)

# Combine all clusters into one array
data_points = np.vstack(all_points)

# Shuffle the points randomly
np.random.shuffle(data_points)

# Create HDF5 file
with h5py.File('/tmp/dataset.h5', 'w') as f:
    # Create a dataset named 'data'
    dset = f.create_dataset('/train', data=data_points, dtype='float32')
    
    # Add attributes to the dataset
    dset.attrs['description'] = 'Random data points with dimension 128'
    dset.attrs['num_points'] = 10000
    dset.attrs['dimension'] = 128
    
print("Dataset created successfully")
