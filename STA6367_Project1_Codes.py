#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 11:45:20 2025

@author: roland
"""


import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from umap import UMAP


import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import adjusted_rand_score




# Set the directory path (replace with your desired path)
directory = '/Users/roland/Desktop/UCF LIBRARY/DATA SCIENCE 2/Project 1 - Clustering/S-Sets'

# Change the working directory
os.chdir(directory)

# Verify that the working directory has changed
print("Current working directory:", os.getcwd())




##============== S-SET ANALYSIS  ==============
# Loading the dataset
s3_dataset_path = '/Users/roland/Desktop/UCF LIBRARY/DATA SCIENCE 2/Project 1 - Clustering/S-Sets/s3.txt'
s4_dataset_path = '/Users/roland/Desktop/UCF LIBRARY/DATA SCIENCE 2/Project 1 - Clustering/S-Sets/s4.txt'

s3_data = pd.read_csv(s3_dataset_path, sep="\s+", header=None, names=['x1', 'x2'])
s4_data = pd.read_csv(s4_dataset_path, sep="\s+", header=None, names=['x1', 'x2'])


# Displaying the first few rows of the dataset
s3_data.head()
s3_data.shape

s4_data.head()
s4_data.shape


## Checking for Missing Values
s3_data.isnull().sum()
s4_data.isnull().sum()


# Standardizing the data
scaler = StandardScaler()
s3_data_scaled = scaler.fit_transform(s3_data)

# Convert the scaled data back to a DataFrame for easier handling in outlier detection
s3_df_scaled = pd.DataFrame(s3_data_scaled,columns=["0","1"])
s3_df_scaled

# Overview of potential outliers by describing the data
s3_df_summary = s3_df_scaled.describe()
s3_df_summary





def plot_dataset(data, title, color, subplot_index):
    plt.subplot(1, 2, subplot_index)
    plt.scatter(data["x1"], data["x2"], s=0.5, color=color)
    plt.title(title, fontsize=40)
    plt.xlabel("x1", fontsize=35)
    plt.ylabel("x2", fontsize=35)
    plt.tick_params(axis='both', which='major', labelsize=28, length=20, width=7)

plt.figure(figsize=(22, 10))

plot_dataset(s3_data, 'S3 Dataset', 'red', 1)
plot_dataset(s4_data, 'S4 Dataset', 'blue', 2)

plt.tight_layout()
plt.show()




### KMEANS
def apply_kmeans(data, n_clusters=15, random_state=561):
    """Applies K-Means clustering to the given dataset and returns the model and cluster labels."""
    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    clusters = model.fit_predict(data)
    # Adjust clusters so they start from 1
    clusters = clusters + 1  # Shift labels to start from 1 instead of 0
    return model, clusters

# Apply K-Means clustering on S3 and S4 dataset
kmeans_model_s3, kmeans_clusters_s3 = apply_kmeans(s3_data)
kmeans_model_s4, kmeans_clusters_s4 = apply_kmeans(s4_data)


#### GMM
def apply_gmm(data, n_components=15, random_state=561):
    """Applies Gaussian Mixture Model clustering and returns the model and cluster labels."""
    model = GaussianMixture(n_components=n_components, random_state=random_state)
    clusters = model.fit_predict(data)
    # Adjust clusters so they start from 1
    clusters = clusters + 1  # Shift labels to start from 1 instead of 0
    return model, clusters

# Apply Gaussian Mixture Model clustering on S3 and S4 dataset
gmm_model_s3, gmm_clusters_s3 = apply_gmm(s3_data)
gmm_model_s4, gmm_clusters_s4 = apply_gmm(s4_data)


## Visualizing Clusters
def visualize_clustering(data, cluster_labels, title, subplot_index):
    """Visualizes clustering results using a scatter plot with a colorbar for clusters."""
    plt.subplot(1, 2, subplot_index)
    scatter = plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=cluster_labels, cmap="viridis", marker=".")
    plt.title(title)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.tick_params(axis='both', which='major', labelsize=15, length=20, width=7)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Cluster Labels', fontsize=12)
    cbar.ax.tick_params(labelsize=10)

# Create a figure for side-by-side visualizations
plt.figure(figsize=(15, 8))

# K-Means Clustering Visualization
visualize_clustering(s3_data, kmeans_clusters_s3, "K-Means Clustering of S3 Set", 1)
visualize_clustering(s4_data, kmeans_clusters_s4, "K-Means Clustering of S4 Set", 2)

plt.tight_layout()
plt.show()



# GMM Clustering Visualization
plt.figure(figsize=(15, 8))
visualize_clustering(s3_data, gmm_clusters_s3, "GMM Clustering of S3 Set", 1)
visualize_clustering(s4_data, gmm_clusters_s4, "GMM Clustering of S4 Set", 2)

plt.tight_layout()
plt.show()





# Load the ground trouth S3 and S4
s3_ground_trouth_path="/Users/roland/Desktop/UCF LIBRARY/DATA SCIENCE 2/Project 1 - Clustering/S-Sets/s3-label.pa"
s4_ground_trouth_path="/Users/roland/Desktop/UCF LIBRARY/DATA SCIENCE 2/Project 1 - Clustering/S-Sets/s4-label.pa"
s3_labels = pd.read_csv(s3_ground_trouth_path,sep="\+",header=None)
s4_labels = pd.read_csv(s4_ground_trouth_path,sep="\+",header=None)


s3_labels.head()


# Extract the labels of S Sets

#S3
s3_labels_final = s3_labels.iloc[5:, 0].reset_index(drop=True).astype(int)
s4_labels_final = s4_labels.iloc[5:, 0].reset_index(drop=True).astype(int)

unique_true_labels_s3 = np.unique(s3_labels_final)
print(unique_true_labels_s3)



# Compute metrics for S3 and S4
def compute_metrics(ground_truth, predicted_labels):
    """Compute ARI and NMI metrics."""
    ari = adjusted_rand_score(ground_truth, predicted_labels)
    nmi = normalized_mutual_info_score(ground_truth, predicted_labels)
    return ari, nmi

# Compute metrics for S3
ari_kmeans_s3, nmi_kmeans_s3 = compute_metrics(s3_labels_final, kmeans_clusters_s3)
ari_gmm_s3, nmi_gmm_s3 = compute_metrics(s3_labels_final, gmm_clusters_s3)

# Compute metrics for S4
ari_kmeans_s4, nmi_kmeans_s4 = compute_metrics(s4_labels_final, kmeans_clusters_s4)
ari_gmm_s4, nmi_gmm_s4 = compute_metrics(s4_labels_final, gmm_clusters_s4)

# Print results for S3
print(f"S3 - KMeans: ARI = {ari_kmeans_s3:.4f}, NMI = {nmi_kmeans_s3:.4f}")
print(f"S3 - GMM: ARI = {ari_gmm_s3:.4f}, NMI = {nmi_gmm_s3:.4f}")

# Print results for S4
print(f"S4 - KMeans: ARI = {ari_kmeans_s4:.4f}, NMI = {nmi_kmeans_s4:.4f}")
print(f"S4 - GMM: ARI = {ari_gmm_s4:.4f}, NMI = {nmi_gmm_s4:.4f}")






#=========== A-SETS ANALYSIS =============
# Loading the dataset
a3_dataset_path = '/Users/roland/Desktop/UCF LIBRARY/DATA SCIENCE 2/Project 1 - Clustering/A-sets/a3.txt'
a3_labels_path = '/Users/roland/Desktop/UCF LIBRARY/DATA SCIENCE 2/Project 1 - Clustering/A-sets/a3-Ground truth partitions.pa'
a3_data = pd.read_csv(a3_dataset_path, sep="\s+", header=None, names=['x1', 'x2'])
a3_labels = pd.read_csv(a3_labels_path, sep="\s+", header=None)
a3_data.shape
a3_data.head()

a3_labels.shape
a3_labels.head()


## EDA
def visualize_dataset(data, title="Dataset Visualization", color="blue", point_size=10):
    """Visualizes a 2D dataset using a scatter plot."""
    #plt.figure(figsize=(10, 8))
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], s=point_size, color=color)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title(title)
    plt.show()

# Visualize A3 dataset
visualize_dataset(a3_data, title="A3 Scatter Plot")



### KMEANS
def apply_kmeans(data, n_clusters=50, random_state=561):
    """Applies K-Means clustering to the given dataset and returns the model and cluster labels."""
    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    clusters = model.fit_predict(data)
    # Adjust clusters so they start from 1
    clusters = clusters + 1  # Shift labels to start from 1 instead of 0
    return model, clusters


# Apply K-Means clustering on A3 dataset
kmeans_model_a3, kmeans_clusters_a3 = apply_kmeans(a3_data)


#### GMM
def apply_gmm(data, n_components=50, random_state=561):
    """Applies Gaussian Mixture Model clustering and returns the model and cluster labels."""
    model = GaussianMixture(n_components=n_components, random_state=random_state)
    clusters = model.fit_predict(data)
    # Adjust clusters so they start from 1
    clusters = clusters + 1  # Shift labels to start from 1 instead of 0
    return model, clusters


# Apply Gaussian Mixture Model clustering on A3 dataset
gmm_model_a3, gmm_clusters_a3 = apply_gmm(a3_data)


## Visualizing Clusters
def visualize_clustering(data, cluster_labels, title, subplot_index):
    """Visualizes clustering results using a scatter plot with a colorbar for clusters."""
    plt.subplot(1, 2, subplot_index)
    scatter = plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=cluster_labels, cmap="viridis", marker=".")
    plt.title(title)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.tick_params(axis='both', which='major', labelsize=15, length=20, width=7)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Cluster Labels', fontsize=12)
    cbar.ax.tick_params(labelsize=10)

# Create a figure for side-by-side visualizations
plt.figure(figsize=(15, 8))


# K-Means Clustering Visualization
visualize_clustering(a3_data, kmeans_clusters_a3, "K-Means Clustering of A3 Set", 1)

# GMM Clustering Visualization
visualize_clustering(a3_data, gmm_clusters_a3, "GMM Clustering of A3 Set", 2)

plt.tight_layout()
plt.show()


# Extract the labels of A3 Set
a3_labels_final=a3_labels.iloc[4:, 0].reset_index(drop=True).astype(int)
a3_labels_final


# Compute metrics for S3
ari_kmeans_a3, nmi_kmeans_a3 = compute_metrics(a3_labels_final, kmeans_clusters_a3)
ari_gmm_a3, nmi_gmm_a3 = compute_metrics(a3_labels_final, gmm_clusters_a3)

print(ari_kmeans_a3,nmi_kmeans_a3)

print(ari_gmm_a3,nmi_gmm_a3)









#=========== BIRCH-SETS ANALYSIS =============
# Loading the dataset
birch1_dataset_path = '/Users/roland/Desktop/UCF LIBRARY/DATA SCIENCE 2/Project 1 - Clustering/Birch-sets/birch1.txt'
birch2_dataset_path = '/Users/roland/Desktop/UCF LIBRARY/DATA SCIENCE 2/Project 1 - Clustering/Birch-sets/birch2.txt'
birch1_data = pd.read_csv(birch1_dataset_path, sep="\s+", header=None, names=['x1', 'x2'])
birch2_data = pd.read_csv(birch2_dataset_path, sep="\s+", header=None, names=['x1', 'x2'])


birch1_data.shape
birch1_data.head()


## EDA
def plot_dataset(data, title, color, subplot_index):
    plt.subplot(1, 2, subplot_index)
    plt.scatter(data["x1"], data["x2"], s=0.5, color=color)
    plt.title(title, fontsize=40)
    plt.xlabel("x1", fontsize=35)
    plt.ylabel("x2", fontsize=35)
    plt.tick_params(axis='both', which='major', labelsize=28, length=20, width=7)

plt.figure(figsize=(25, 13))

plot_dataset(birch1_data, 'Birch-1 Dataset', 'red', 1)
plot_dataset(birch2_data, 'Birch-2 Dataset', 'blue', 2)

plt.tight_layout()
plt.show()



# Applying K-means/GMM clustering on Birch sets

### KMEANS
def apply_kmeans(data, n_clusters=100, random_state=561):
    """Applies K-Means clustering to the given dataset and returns the model and cluster labels."""
    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    clusters = model.fit_predict(data)
    # Adjust clusters so they start from 1
    clusters = clusters + 1  # Shift labels to start from 1 instead of 0
    return model, clusters

# Apply K-Means clustering on A3 dataset
kmeans_model_birch1, kmeans_clusters_birch1 = apply_kmeans(birch1_data, n_clusters=100, random_state=561)
kmeans_model_birch2, kmeans_clusters_birch2 = apply_kmeans(birch2_data, n_clusters=100, random_state=561)


#### GMM
def apply_gmm(data, n_components=100, random_state=561):
    """Applies Gaussian Mixture Model clustering and returns the model and cluster labels."""
    model = GaussianMixture(n_components=n_components, random_state=random_state)
    clusters = model.fit_predict(data)
    # Adjust clusters so they start from 1
    clusters = clusters + 1  # Shift labels to start from 1 instead of 0
    return model, clusters

# Apply Gaussian Mixture Model clustering on A3 dataset
gmm_model_birch1, gmm_clusters_birch1 = apply_gmm(birch1_data, n_components=100, random_state=561)
gmm_model_birch2, gmm_clusters_birch2 = apply_gmm(birch2_data, n_components=100, random_state=561)



#### Visualizing Clusters
def visualize_clustering(data, cluster_labels, title, subplot_index):
    """Visualizes clustering results using a scatter plot with a colorbar for clusters."""
    plt.subplot(1, 2, subplot_index)
    scatter = plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=cluster_labels, cmap="viridis", marker=".")
    plt.title(title)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.tick_params(axis='both', which='major', labelsize=15, length=20, width=7)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Cluster Labels', fontsize=12)
    cbar.ax.tick_params(labelsize=10)

# Create a figure for side-by-side visualizations
plt.figure(figsize=(15, 8))


# K-Means Clustering Visualization
visualize_clustering(birch1_data, kmeans_clusters_birch1, "K-Means Clustering of Birch-1 Set", 1)
visualize_clustering(birch2_data, kmeans_clusters_birch2, "K-Means Clustering of Birch-2 Set", 2)
plt.tight_layout()
plt.show()



# GMM Clustering Visualization
plt.figure(figsize=(15, 8))
visualize_clustering(birch1_data, gmm_clusters_birch1, "GMM Clustering of Birch-1 Set", 1)
visualize_clustering(birch2_data, gmm_clusters_birch2, "GMM Clustering of Birch-2 Set", 2)
plt.tight_layout()
plt.show()



### Importing Labels
birch1_labels_path = '/Users/roland/Desktop/UCF LIBRARY/DATA SCIENCE 2/Project 1 - Clustering/Birch-sets/b1-gt.pa'
birch2_labels_path = '/Users/roland/Desktop/UCF LIBRARY/DATA SCIENCE 2/Project 1 - Clustering/Birch-sets/b2-gt.pa'
birch1_labels = pd.read_csv(birch1_labels_path, sep="\s+", header=None)
birch2_labels = pd.read_csv(birch2_labels_path, sep="\s+", header=None)


# Extracting the labels of Birch Set
birch1_labels_final = birch1_labels.iloc[4:, 0].reset_index(drop=True).astype(int)
birch1_labels_final

birch2_labels_final = birch2_labels.iloc[4:, 0].reset_index(drop=True).astype(int)
birch2_labels_final


# Compute metrics for Birch Sets
ari_kmeans_birch1, nmi_kmeans_birch1 = compute_metrics(birch1_labels_final, kmeans_clusters_birch1)
ari_kmeans_birch2, nmi_kmeans_birch2 = compute_metrics(birch2_labels_final, kmeans_clusters_birch2)
print(ari_kmeans_birch1, nmi_kmeans_birch1)
print(ari_kmeans_birch2, nmi_kmeans_birch2)

ari_gmm_birch1, nmi_gmm_birch1 = compute_metrics(birch1_labels_final, gmm_clusters_birch1)
ari_gmm_birch2, nmi_gmm_birch2 = compute_metrics(birch2_labels_final, gmm_clusters_birch2)
print(ari_gmm_birch1, nmi_gmm_birch1)
print(ari_gmm_birch2, nmi_gmm_birch2)



np.unique(kmeans_clusters_birch1)
np.unique(birch1_labels_final)







#=========== G2-SETS ANALYSIS =============
# Loading the dataset
g2_dataset_path = '/Users/roland/Desktop/UCF LIBRARY/DATA SCIENCE 2/Project 1 - Clustering/G2-sets/g2-1024-100.txt'
g2_data = pd.read_csv(g2_dataset_path, sep="\s+", header=None)
g2_data.shape
g2_data.head()


# Dimension Reduction with UMAP
umap_model = UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
g2_data_umap = umap_model.fit_transform(g2_data)
g2_data_umap


def visualize_dataset(data, title="Dataset Visualization", color="blue", point_size=10):
    """Visualizes a 2D dataset using a scatter plot."""
    plt.figure(figsize=(10, 7))
    plt.scatter(data[:, 0], data[:, 1], s=point_size, color=color)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title(title)
    plt.show()

# Visualize A3 dataset
visualize_dataset(g2_data_umap, title="G2 Dataset")


# Applying K-means/GMM clustering on G2 sets

### KMEANS
def apply_kmeans(data, n_clusters=2, random_state=561):
    """Applies K-Means clustering to the given dataset and returns the model and cluster labels."""
    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    clusters = model.fit_predict(data)
    # Adjust clusters so they start from 1
    clusters = clusters + 1  # Shift labels to start from 1 instead of 0
    return model, clusters

# Apply K-Means clustering on A3 dataset
kmeans_model_g2, kmeans_clusters_g2 = apply_kmeans(g2_data_umap, n_clusters=2, random_state=561)


#### GMM
def apply_gmm(data, n_components=2, random_state=561):
    """Applies Gaussian Mixture Model clustering and returns the model and cluster labels."""
    model = GaussianMixture(n_components=n_components, random_state=random_state)
    clusters = model.fit_predict(data)
    # Adjust clusters so they start from 1
    clusters = clusters + 1  # Shift labels to start from 1 instead of 0
    return model, clusters

# Apply Gaussian Mixture Model clustering on A3 dataset
gmm_model_g2, gmm_clusters_g2 = apply_gmm(g2_data_umap, n_components=2, random_state=561)


#### Visualizing Clusters
def visualize_clustering(data, cluster_labels, title, subplot_index):
    """Visualizes clustering results using a scatter plot with a colorbar for clusters."""
    plt.subplot(1, 2, subplot_index)
    scatter = plt.scatter(data[:, 0], data[:, 1], c=cluster_labels, cmap="viridis", marker=".")
    plt.title(title)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.tick_params(axis='both', which='major', labelsize=15, length=20, width=7)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Cluster Labels', fontsize=12)
    cbar.ax.tick_params(labelsize=10)

# Create a figure for side-by-side visualizations
plt.figure(figsize=(15, 8))

# K-Means Clustering Visualization
visualize_clustering(g2_data_umap, kmeans_clusters_g2, "K-Means Clustering of G2 Set", 1)
visualize_clustering(g2_data_umap, gmm_clusters_g2, "GMM Clustering of G2 Set", 2)
plt.tight_layout()
plt.show()




g2_labels_path = '/Users/roland/Desktop/UCF LIBRARY/DATA SCIENCE 2/Project 1 - Clustering/G2-sets/g2-1024-100-round truth partitions.pa'
g2_labels = pd.read_csv(g2_labels_path, sep="\s+", header=None)



# Extract the labels of A3 Set
g2_labels_final = g2_labels.iloc[4:, 0].reset_index(drop=True).astype(int)
g2_labels_final



# Compute metrics for g2
ari_kmeans_g2, nmi_kmeans_g2 = compute_metrics(g2_labels_final, kmeans_clusters_g2)
ari_gmm_g2, nmi_gmm_g2 = compute_metrics(g2_labels_final, gmm_clusters_g2)

print(ari_kmeans_g2,nmi_kmeans_g2)

print(ari_gmm_g2,nmi_gmm_g2)






#=========== DIM-SETS ANALYSIS =============
# Loading the dataset
d1_dataset_path = '/Users/roland/Desktop/UCF LIBRARY/DATA SCIENCE 2/Project 1 - Clustering/DIM-sets/dim032.txt'
d2_dataset_path = '/Users/roland/Desktop/UCF LIBRARY/DATA SCIENCE 2/Project 1 - Clustering/DIM-sets/dim1024.txt'
d1_data = pd.read_csv(d1_dataset_path, sep="\s+", header=None)
d2_data = pd.read_csv(d2_dataset_path, sep="\s+", header=None)
d1_data.shape
d1_data.head()



### Dimension Reduction with UMAP
def apply_umap(data, n_neighbors=15, min_dist=0.1, n_components=2):
    """Applies UMAP dimensionality reduction on the given dataset."""
    umap_model = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components)
    return umap_model.fit_transform(data)

# Apply UMAP on DIM sets
d1_data_umap = apply_umap(d1_data)
d2_data_umap = apply_umap(d2_data)


### Dim Set Visualization
def plot_dataset(data, title, color, subplot_index):
    plt.subplot(1, 2, subplot_index)
    plt.scatter(data[:, 0], data[:, 1], s=0.5, color=color)
    plt.title(title, fontsize=40)
    plt.xlabel("UMAP 1", fontsize=35)
    plt.ylabel("UMAP 2", fontsize=35)
    plt.tick_params(axis='both', which='major', labelsize=28, length=20, width=7)

plt.figure(figsize=(18, 10))

plot_dataset(d1_data_umap, 'Dim032 Dataset', 'red', 1)
plot_dataset(d2_data_umap, 'Dim1024 Dataset', 'blue', 2)

plt.tight_layout()
plt.show()




# Applying K-means/GMM clustering on G2 sets

### KMEANS
def apply_kmeans(data, n_clusters=16, random_state=561):
    """Applies K-Means clustering to the given dataset and returns the model and cluster labels."""
    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    clusters = model.fit_predict(data)
    # Adjust clusters so they start from 1
    clusters = clusters + 1  # Shift labels to start from 1 instead of 0
    return model, clusters

# Apply K-Means clustering on A3 dataset
kmeans_model_d1, kmeans_clusters_d1 = apply_kmeans(d1_data_umap, n_clusters=16, random_state=561)
kmeans_model_d2, kmeans_clusters_d2 = apply_kmeans(d2_data_umap, n_clusters=16, random_state=561)


#### GMM
def apply_gmm(data, n_components=16, random_state=561):
    """Applies Gaussian Mixture Model clustering and returns the model and cluster labels."""
    model = GaussianMixture(n_components=n_components, random_state=random_state)
    clusters = model.fit_predict(data)
    # Adjust clusters so they start from 1
    clusters = clusters + 1  # Shift labels to start from 1 instead of 0
    return model, clusters

# Apply Gaussian Mixture Model clustering on A3 dataset
gmm_model_d1, gmm_clusters_d1 = apply_gmm(d1_data_umap, n_components=16, random_state=561)
gmm_model_d2, gmm_clusters_d2 = apply_gmm(d2_data_umap, n_components=16, random_state=561)



#### Visualizing Clusters
def visualize_clustering(data, cluster_labels, title, subplot_index):
    """Visualizes clustering results using a scatter plot with a colorbar for clusters."""
    plt.subplot(1, 2, subplot_index)
    scatter = plt.scatter(data[:, 0], data[:, 1], c=cluster_labels, cmap="viridis", marker=".")
    plt.title(title)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.tick_params(axis='both', which='major', labelsize=15, length=20, width=7)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Cluster Labels', fontsize=12)
    cbar.ax.tick_params(labelsize=10)

# Create a figure for side-by-side visualizations
plt.figure(figsize=(15, 8))

# K-Means Clustering Visualization
visualize_clustering(d1_data_umap, kmeans_clusters_d1, "K-Means Clustering of Dim032 Set", 1)
visualize_clustering(d2_data_umap, kmeans_clusters_d2, "K-Means Clustering of Dim1024 Set", 2)
plt.tight_layout()
plt.show()


# KMeans and GMM Clustering Visualization
plt.figure(figsize=(15, 8))
visualize_clustering(d1_data_umap, gmm_clusters_d1, "GMM Clustering of Dim1024 Set", 1)
visualize_clustering(d2_data_umap, gmm_clusters_d2, "GMM Clustering of Dim1024 Set", 2)
plt.tight_layout()
plt.show()



### Importing Labels
d1_labels_path = '/Users/roland/Desktop/UCF LIBRARY/DATA SCIENCE 2/Project 1 - Clustering/DIM-sets/dim032.pa'
d2_labels_path = '/Users/roland/Desktop/UCF LIBRARY/DATA SCIENCE 2/Project 1 - Clustering/DIM-sets/dim1024.pa'
d1_labels = pd.read_csv(d1_labels_path, sep="\s+", header=None)
d2_labels = pd.read_csv(d2_labels_path, sep="\s+", header=None)
d1_labels.shape
d1_labels.head()



# Extract the labels of A3 Set
d1_labels_final = d1_labels.iloc[5:, 0].reset_index(drop=True).astype(int)
d1_labels_final

d2_labels_final = d2_labels.iloc[5:, 0].reset_index(drop=True).astype(int)
d2_labels_final


# Compute metrics for g2
ari_kmeans_d1, nmi_kmeans_d1 = compute_metrics(d1_labels_final, kmeans_clusters_d1)
ari_gmm_d1, nmi_gmm_d1 = compute_metrics(d1_labels_final, gmm_clusters_d1)

ari_kmeans_d2, nmi_kmeans_d2 = compute_metrics(d2_labels_final, kmeans_clusters_d2)
ari_gmm_d2, nmi_gmm_d2 = compute_metrics(d2_labels_final, gmm_clusters_d2)





#=========== UNBALANCED-SETS ANALYSIS =============
# Loading the dataset
unbalance_dataset_path = '/Users/roland/Desktop/UCF LIBRARY/DATA SCIENCE 2/Project 1 - Clustering/Unbalance-sets/unbalance.txt'
unbalance_data = pd.read_csv(unbalance_dataset_path, sep="\s+", header=None, names=['x1', 'x2'])
unbalance_data.shape
unbalance_data.head()


### EDA
def visualize_dataset(data, title="Dataset Visualization", color="blue", point_size=10):
    """Visualizes a 2D dataset using a scatter plot."""
    plt.figure(figsize=(15, 10))
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], s=point_size, color=color)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title(title)
    plt.show()

# Visualize Unbalance Set dataset
visualize_dataset(unbalance_data, title="Unbalance Dataset")


### KMEANS
def apply_kmeans(data, n_clusters=8, random_state=561):
    """Applies K-Means clustering to the given dataset and returns the model and cluster labels."""
    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    clusters = model.fit_predict(data)
    # Adjust clusters so they start from 1
    clusters = clusters + 1  # Shift labels to start from 1 instead of 0
    return model, clusters

# Apply K-Means clustering on Unbalance Set
kmeans_model_unbalance, kmeans_clusters_unbalance = apply_kmeans(unbalance_data, n_clusters=8, random_state=561)


#### GMM
def apply_gmm(data, n_components=8, random_state=561):
    """Applies Gaussian Mixture Model clustering and returns the model and cluster labels."""
    model = GaussianMixture(n_components=n_components, random_state=random_state)
    clusters = model.fit_predict(data)
    # Adjust clusters so they start from 1
    clusters = clusters + 1  # Shift labels to start from 1 instead of 0
    return model, clusters

# Apply Gaussian Mixture Model clustering on A3 dataset
gmm_model_unbalance, gmm_clusters_unbalance = apply_gmm(unbalance_data, n_components=8, random_state=561)





#### Visualizing Clusters
def visualize_clustering(data, cluster_labels, title, subplot_index):
    """Visualizes clustering results using a scatter plot with a colorbar for clusters."""
    plt.subplot(1, 2, subplot_index)
    scatter = plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=cluster_labels, cmap="viridis", marker=".")
    plt.title(title)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.tick_params(axis='both', which='major', labelsize=15, length=20, width=7)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Cluster Labels', fontsize=12)
    cbar.ax.tick_params(labelsize=10)

# Create a figure for side-by-side visualizations
plt.figure(figsize=(15, 8))

# K-Means Clustering Visualization
visualize_clustering(unbalance_data, kmeans_clusters_unbalance, "K-Means Clustering of Unbalance Set", 1)

# GMM Clustering Visualization
visualize_clustering(unbalance_data, gmm_clusters_unbalance, "GMM Clustering of Unbalance Set", 2)

plt.tight_layout()
plt.show()





### Importing Labels
unbalance_labels_path = '/Users/roland/Desktop/UCF LIBRARY/DATA SCIENCE 2/Project 1 - Clustering/Unbalance-sets/unbalance-gt.pa'
unbalance_labels = pd.read_csv(unbalance_labels_path, sep="\s+", header=None)
unbalance_labels.shape
unbalance_labels.head()





# Extract the labels of Unbalance Set
unbalance_labels_final = unbalance_labels.iloc[4:, 0].reset_index(drop=True).astype(int)
unbalance_labels_final



# Compute metrics for Unbalance Set
ari_kmeans_unbalance, nmi_kmeans_unbalance = compute_metrics(unbalance_labels_final, kmeans_clusters_unbalance)
ari_gmm_unbalance, nmi_gmm_unbalance = compute_metrics(unbalance_labels_final, gmm_clusters_unbalance)

























