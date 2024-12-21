# Fashion Image Clustering

This Python package enables the clustering of fashion images into descriptive categories and classifies new images from URLs based on their similarity to existing clusters.

## Disclaimer
The package is not tested in other os than windows, neither on a vm, only to an empty venv.

## Overview
The fashion_image_clustering package uses a pre-trained ResNet model to extract features from images, clusters these features, and generates descriptive labels for each cluster. The user can also classify new images by providing their URLs. If a new image is similar to an existing cluster, the package assigns it to that cluster; otherwise, it marks the image as an outlier.

## Features

### Clustering of Image Dataset:

* Preprocesses a folder of images (e.g., fashion images) and extracts features.
* Clusters the images and assigns descriptive labels to each cluster.
* Allows the creation of unique identifiers for each clustered dataset.

### Classifying New Images:

* Accepts image URLs as input.
* Classifies the image into an existing cluster or identifies it as an outlier.

## Installation

### Step 1: Set up the Environment

The package requires Python 3.8 or newer. Use the provided venv directory structure:

1.Navigate to the project root and activate the virtual environment:

> .\venv\Scripts\Activate  # For Windows

Choose the venv interpreter for venv for python.exe.

2. Install the package:
> pip install numpy torch torchvision opencv-python scikit-learn requests joblib pillow

### Step 2: Prepare the Dataset
Place the images you want to cluster into a folder within the data directory, and adjust data_dir in main.py to point to this folder (e.g. the existing dataset data/fashion485).

## Usage

### Running the Main Script

The main script (main.py) provides an interactive interface for clustering and classification.

>python src/clustering_20241108/main.py

Upon running the script, follow the prompts to either:

1. Train a New Clustering Model:

* You will be prompted to provide a unique identifier for the dataset (e.g., fashion485_v1).
* This identifier helps you reference the dataset later when classifying new images.

2. Classify an Image from a URL:

* Choose an identifier for the clustering model you wish to use.
* Enter the image URL. The script will preprocess and classify the image, returning either a cluster label or marking the image as an outlier.

3. exit 

### Example Console Interaction
Do you want to train a new clustering model or classify an image? (train/classify/exit): train

Enter a unique identifier for this dataset (e.g., fashion485_v1): fashion485_v2

Extracting features from dataset...

Cluster Descriptive Labels: {0: 'gown/overskirt', 1: 'gown/sarong', ... }
Model saved with identifier: fashion485_v2

Do you want to train a new clustering model or classify an image? (train/classify/exit): classify

Available identifiers: ['fashion485_v1', 'fashion485_v2']

Enter the identifier of the model you wish to use for classification: fashion485_v2

Enter the URL of the image to classify: https://www.sandalworkshop.gr/wp-content/uploads/2021/05/Santorini-mens-handame-sandals-in-black-leather_main-pic.jpg

Cluster label: buckle/sandal


## Directory Structure

The following structure organizes the package for development and usage:


fashion_image_clustering/
├── data/                            # Store dataset images and clustering models
│   ├── fashion485/                  # Original image dataset directory
│   └── saved_models/                # Directory for saving clustering model files
│       └── <model_identifier>/      # Folder for each unique model identifier
│           ├── centroids.npz        # Saved centroids of clusters
│           ├── cluster_labels.npy   # Array of cluster labels
│           ├── clustering_model.txt # Descriptive labels for each cluster
│           └── scaler.npz           # Scaler parameters for feature normalization
├── notebooks/                       # Jupyter notebooks for experimentation
│   ├── DL_model/                    # Deep learning approaches the pytorch approach is the final one  
│   └── vanilla/                     # Clustering method experiments 
├── src/clustering_20241108/         # Source code for the clustering package
│   ├── __init__.py                  # Package initialization
│   ├── clustering.py                # Clustering model
│   ├── data_preprocessor.py         # Image preprocessing functions
│   ├── feature_extractor.py         # Feature extraction
│   ├── image_downloader.py          # Downloading images from URLs
│   ├── main.py                      # Main script for clustering and classification
│   └── url_cluster_infer.py         # URL-based image classification
├── venv/                            # Python virtual environment
├── LICENSE                          # License file
├── pyproject.toml                   # Project configuration for dependencies
└── README.md                        # Project documentation

## Module Guide

### Data Preprocessor (data_preprocessor.py)
* resize_with_aspect_ratio: Resizes images with aspect ratio preservation.
* apply_clahe: Enhances image contrast using CLAHE.
* remove_colored_bubbles: Removes specific colored artifacts based on HSV filtering.
* apply_edge_detection: Applies edge detection for feature enhancement.
* preprocess_image: Calls all preprocessing functions to prepare an image.


### Feature Extractor (feature_extractor.py)
Extracts features using a pre-trained ResNet model.
* extract_features: Extracts features from all images in a directory.
* extract_features_from_image: Extracts features from a single image.
* save_scaler & load_scaler: Save and load the scaler for standardizing features.

### Clustering Model (clustering.py)
Clusters features and assigns descriptive labels.
* fit: Clusters extracted features and calculates centroids.
* generate_labels_from_predictions: Assigns descriptive labels to clusters.
* save_model_and_labels & load_model_and_labels: Save and load clustering information.

### Image Downloader (image_downloader.py)
Downloads images from URLs with retry logic and preprocesses them.

### URL Cluster Inference (url_cluster_infer.py)
Classifies new images by comparing them to cluster centroids.
* load_model_for_identifier: Loads model and scaler based on dataset identifier.
* cosine_similarity: Calculates similarity between features and centroids.
* classify_image_from_url: Classifies a new image from a URL.

## Configuration Guide

### Clustering New Datasets
1. Add Images: Place new images in the data directory (e.g., data/new_dataset/).
2. Update data_dir: Modify data_dir in main.py to point to the new dataset.
3. Train a New Model: Run the main script and choose the "train" option to cluster the new dataset and save it with a unique identifier.

### Preprocessing Options
The preprocessing steps are configured in data_preprocessor.py:
* resize_with_aspect_ratio: Resizes images with aspect ratio preservation.
* apply_clahe: Enhances image contrast using CLAHE.
* remove_colored_bubbles: Removes specific artifacts.
* apply_edge_detection: Applies edge detection for feature enhancement.
* preprocess_image: Combines all preprocessing steps.

These functions can be enabled or modified to suit different datasets. Adjust options by setting parameters like use_clahe, mask_bubbles, or use_edge_detection in the DataPreprocessor class in main.py.

### Feature Extraction Parameters
The feature_extractor.py module defines feature extraction settings. Modify these only if you change the base model or clustering approach. Key functions include:
* extract_features: Extracts features for clustering.
* extract_features_from_image: Extracts features from a single image URL.
* save_scaler & load_scaler: Save and load feature scaling parameters.

## Dependencies
Dependencies are managed in pyproject.toml

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
For questions, contact Nikos Lavdakis at lavdisn@gmail.com.