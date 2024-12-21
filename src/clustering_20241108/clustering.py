# clustering.py
import os
import random
import numpy as np
import torch
from collections import Counter
from sklearn.cluster import AgglomerativeClustering
from PIL import Image
from collections import defaultdict

class ClusteringModel:
    def __init__(self, n_clusters=15, linkage="ward"):
        """
        Initializes the clustering model.
        :param n_clusters: Number of clusters for Agglomerative Clustering.
        :param linkage: Linkage criterion for clustering.
        """
        self.model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        self.cluster_labels_dict = {}  # Dictionary to store labels for each cluster
        self.labels = None  # Placeholder for the labels of fitted features
        self.centroids = {}  # Dictionary to store centroids of clusters


    def fit(self, features):
        """
        Fits the clustering model on the extracted features and calculates centroids.
        :param features: Array of feature vectors to cluster.
        :return: Cluster labels for each feature vector.
        """
        # Perform clustering
        self.labels = self.model.fit_predict(features)

        # Calculate centroids for each cluster after fitting
        unique_labels = set(self.labels)  # Only iterate over labels after they are assigned
        self.centroids = {
            label: features[self.labels == label].mean(axis=0)
            for label in unique_labels if label != -1  # Exclude noise points if any
        }

        return self.labels

    # These parameters cannot be defined in main so you adjust them on your dataset only from here
    def generate_labels_from_predictions(self, cluster_labels, img_names, img_dir, feature_extractor, n_samples=32,
                                         top_n=6):
        """
        Generates descriptive labels for each cluster based on image content predictions.
        :param cluster_labels: Array of cluster labels for each image.
        :param img_names: List of image filenames corresponding to cluster_labels.
        :param img_dir: Directory containing the images.
        :param feature_extractor: FeatureExtractor instance to predict image labels.
        :param n_samples: Number of images to sample per cluster for label generation.
        :param top_n: Number of top predictions to consider for generating cluster labels.
        :return: Dictionary of descriptive labels for each cluster.
        """
        cluster_label_counts = defaultdict(Counter)

        for cluster in range(max(cluster_labels) + 1):
            cluster_indices = np.where(cluster_labels == cluster)[0]
            sample_indices = random.sample(list(cluster_indices), min(n_samples, len(cluster_indices)))

            for idx in sample_indices:
                img_path = os.path.join(img_dir, img_names[idx])
                img = Image.open(img_path).convert("RGB")
                img_tensor = feature_extractor.preprocess(img).unsqueeze(0).to(feature_extractor.device)

                with torch.no_grad():
                    preds = feature_extractor.model(img_tensor)
                    top_predictions = torch.topk(preds, top_n).indices.squeeze(0).tolist()
                    predicted_labels = [feature_extractor.get_label_name(i) for i in top_predictions]

                    # Update the label count for this cluster
                    cluster_label_counts[cluster].update(predicted_labels)

            # Use the most common labels as the cluster label
            # You can add conditions to make sure clusters have unique labels if needed
            common_labels = [label for label, _ in cluster_label_counts[cluster].most_common(2)]
            cluster_label = "/".join(common_labels)
            self.cluster_labels_dict[cluster] = cluster_label

        return self.cluster_labels_dict

    def generate_cluster_labels(self, img_names, img_dir, feature_extractor):
        """
        Main function to generate descriptive labels for clusters after fitting.
        :param img_names: List of image names.
        :param img_dir: Directory of images for label generation.
        :param feature_extractor: The feature extractor to predict image labels.
        :return: Dictionary of cluster labels.
        """
        if self.labels is None:
            raise ValueError("Clustering model must be fitted before generating labels.")

        return self.generate_labels_from_predictions(self.labels, img_names, img_dir, feature_extractor)

    def save_model_and_labels(self, model_path, labels_path, centroids_path):
        """
        Saves the clustering model's labels, centroids, and cluster label dictionary.

        :param model_path: Path to save the dictionary containing descriptive labels for each cluster.
        :param labels_path: Path to save the array of cluster labels for each data point.
        :param centroids_path: Path to save the centroids for each cluster.
            The centroids are saved as individual arrays in an .npz file, where each array is labeled by the cluster index.
        """
        np.save(labels_path, self.labels)

        # Convert integer keys to strings for compatibility with np.savez
        centroids_with_str_keys = {str(label): centroid for label, centroid in self.centroids.items()}
        np.savez(centroids_path, **centroids_with_str_keys)

        with open(model_path, "w") as f:
            f.write(str(self.cluster_labels_dict))

    def load_model_and_labels(self, model_path, labels_path, centroids_path):
        """
        Loads the clustering model's labels, centroids, and cluster label dictionary.

        :param model_path: Path to the saved cluster label dictionary containing descriptive labels for each cluster.
        :param labels_path: Path to the saved array of cluster labels for each data point.
        :param centroids_path: Path to the saved centroids for each cluster.
            The centroids are loaded from an .npz file, where each array is labeled by the cluster index.
        """
        self.labels = np.load(labels_path)
        # Convert string keys back to integers when loading
        self.centroids = {int(label): centroid for label, centroid in np.load(centroids_path).items()}

        with open(model_path, "r") as f:
            self.cluster_labels_dict = eval(f.read())

