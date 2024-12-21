# main.py
import os
import torch
import json
import urllib.request
from torchvision import models, transforms
from data_preprocessor import DataPreprocessor
from feature_extractor import FeatureExtractor
from clustering import ClusteringModel
from image_downloader import ImageDownloader
from url_cluster_infer import URLClusterInference

def main():
    # Paths and model setup
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_dir = os.path.join(project_root, "data", "fashion485")  # Directory of the clustering data
    saved_models_dir = os.path.join(project_root, "data", "saved_models")
    os.makedirs(saved_models_dir, exist_ok=True)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load ImageNet labels and ResNet model used in order to decode the "encrypted" labels of the feature extractor
    labels_url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    with urllib.request.urlopen(labels_url) as f:
        imagenet_labels = {i: label for i, label in enumerate(json.load(f))}
    resnet_model = models.resnet50(pretrained=True).eval().to(device)

    # Normallization for the images
    normalize_preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Initialize components used next in the processing clustering downloading etc
    preprocessor = DataPreprocessor(target_size=(224, 224), use_clahe=True, mask_bubbles=False, use_edge_detection=False) # Functions explained on preprocessor
    feature_extractor = FeatureExtractor(resnet_model, normalize_preprocess, device, label_map=imagenet_labels)
    clustering_model = ClusteringModel(n_clusters=14)
    downloader = ImageDownloader(preprocessor)

    # URL-based classification loop for the users to define the procedure desired
    while True:
        user_input = input("Do you want to train a new clustering model or classify an image? (train/classify/exit): ").strip().lower()

        if user_input == "exit":
            break

        # Training for a new identifer
        elif user_input == "train":
            # Get identifier for the dataset from the user
            dataset_identifier = input("Enter a unique identifier for this dataset (e.g., fashion485_v1): ").strip()
            identifier_dir = os.path.join(saved_models_dir, dataset_identifier)
            os.makedirs(identifier_dir, exist_ok=True)

            # Step 1: Extract dataset features and calculate mean, std
            print("Extracting features from dataset...")
            features, img_names = feature_extractor.extract_features(data_dir)
            dataset_mean = features.mean(axis=0)
            dataset_std = features.std(axis=0)

            # Step 2: Clustering
            labels = clustering_model.fit(features)
            cluster_labels_dict = clustering_model.generate_labels_from_predictions(labels, img_names, data_dir, feature_extractor)
            print("Cluster Descriptive Labels:", cluster_labels_dict)

            # Save model, labels, centroids, and scaler using identifier
            clustering_model.save_model_and_labels(
                os.path.join(identifier_dir, "clustering_model.txt"),
                os.path.join(identifier_dir, "cluster_labels.npy"),
                os.path.join(identifier_dir, "centroids.npz")
            )
            feature_extractor.save_scaler(os.path.join(identifier_dir, "scaler.npz"))

            print(f"Model saved with identifier: {dataset_identifier}")

        # Classify url in existing cluster model dataset
        elif user_input == "classify":
            # List available identifiers
            available_identifiers = os.listdir(saved_models_dir)
            if not available_identifiers:
                print("No saved models available. Please train a model first.")
                continue

            print("Available identifiers:", available_identifiers)
            chosen_identifier = input("Enter the identifier of the model you wish to use for classification: ").strip()

            # Validate the chosen identifier
            if chosen_identifier not in available_identifiers:
                print(f"Identifier '{chosen_identifier}' not found. Please choose a valid identifier.")
                continue

            identifier_dir = os.path.join(saved_models_dir, chosen_identifier)

            # Initialize URLClusterInference and load the corresponding model
            url_inference = URLClusterInference(
                feature_extractor=feature_extractor,
                clustering_model=clustering_model,
                downloader=downloader,
                outlier_threshold=0.4, # Define threshold if needed
                preprocessor=preprocessor
            )
            url_inference.load_model_for_identifier(identifier_dir)

            # URL-based classification
            image_url = input("Enter the URL of the image to classify: ")
            result = url_inference.classify_image_from_url(image_url)
            print(result)

if __name__ == "__main__":
    main()
