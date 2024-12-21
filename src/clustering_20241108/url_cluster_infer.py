# url_cluster_infer.py
import os
import numpy as np
from feature_extractor import FeatureExtractor
from clustering import ClusteringModel
from image_downloader import ImageDownloader
from data_preprocessor import DataPreprocessor

class URLClusterInference:
    def __init__(self, feature_extractor: FeatureExtractor, clustering_model: ClusteringModel,
                 downloader: ImageDownloader, outlier_threshold=0.5, dataset_mean=None, dataset_std=None,
                 preprocessor: DataPreprocessor = None):
        self.feature_extractor = feature_extractor
        self.clustering_model = clustering_model
        self.downloader = downloader
        self.outlier_threshold = outlier_threshold
        self.dataset_mean = dataset_mean
        self.dataset_std = dataset_std
        self.preprocessor = preprocessor

    def load_model_for_identifier(self, identifier_dir):
        """Load model, scaler, and clustering information based on the dataset identifier."""
        # Load clustering model and labels
        self.clustering_model.load_model_and_labels(
            os.path.join(identifier_dir, "clustering_model.txt"),
            os.path.join(identifier_dir, "cluster_labels.npy"),
            os.path.join(identifier_dir, "centroids.npz")
        )
        # Load scaler
        self.feature_extractor.load_scaler(os.path.join(identifier_dir, "scaler.npz"))

    def cosine_similarity(self, a, b):
        """Compute cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    # Euclidean doesnt work that well but you can try it for other data sets
    def classify_image_from_url(self, url):
        # Step 1: Download and preprocess the image
        image = self.downloader.download_image(url)
        if image is None:
            return "Failed to download or preprocess image."

        if self.preprocessor is not None:
            image = self.preprocessor.preprocess_image(image)

        # Step 2: Extract and normalize features
        feature_vector = self.feature_extractor.extract_features_from_image(image)
        if self.dataset_mean is not None and self.dataset_std is not None:
            feature_vector = (feature_vector - self.dataset_mean) / self.dataset_std

        # Step 3: Calculate cosine similarity to each cluster's centroid
        best_similarity = -1
        closest_cluster = None
        for label, centroid in self.clustering_model.centroids.items():
            similarity = self.cosine_similarity(feature_vector, centroid)
            if similarity > best_similarity:
                best_similarity = similarity
                closest_cluster = label

        # Step 4: Apply threshold to determine if it belongs to a cluster or is an outlier
        print(f"Best similarity: {best_similarity} Threshold: {self.outlier_threshold}")
        if best_similarity >= self.outlier_threshold:
            return f"Cluster label: {self.clustering_model.cluster_labels_dict.get(closest_cluster, 'Unknown Cluster')}"
        else:
            return "The image is classified as an outlier."
