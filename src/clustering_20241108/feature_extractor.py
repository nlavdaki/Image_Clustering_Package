# feature_extractor.py
import os
import numpy as np
import torch
from PIL import Image
from sklearn.preprocessing import StandardScaler

class FeatureExtractor:
    def __init__(self, model, preprocess, device="cpu", label_map=None):
        """
        Initializes the feature extractor.
        :param model: A pre-trained model used for feature extraction.
        :param preprocess: Preprocessing transformations to apply to each image.
        :param device: Device on which the model will run.
        :param label_map: Optional dictionary mapping model output indices to human-readable labels.
        """
        self.model = model.to(device)
        self.preprocess = preprocess
        self.device = device
        self.scaler = StandardScaler()
        self.label_map = label_map  # Dictionary mapping model output indices to label names

    # Extracting features in order to identify cluster names later
    def extract_features(self, img_dir):
        """
        Extracts and standardizes features for each image in the specified directory.
        :param img_dir: Directory containing images to process.
        :return: Tuple of standardized feature array and list of image names.
        """
        features = []
        img_names = []

        for img_file in os.listdir(img_dir):
            img_path = os.path.join(img_dir, img_file)
            img = Image.open(img_path).convert("RGB")
            img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                feature = self.model(img_tensor).cpu().numpy().squeeze()
                features.append(feature)
                img_names.append(img_file)

        features = np.array(features)
        standardized_features = self.scaler.fit_transform(features)
        return standardized_features, img_names

    def extract_features_from_image(self, image):
        """
        Extracts features from a single PIL image (for real-time inference).
        :param image: A preprocessed PIL image.
        :return: Standardized feature vector for the image.
        """
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feature = self.model(img_tensor).cpu().numpy().squeeze()
        # Return the standardized feature vector
        return self.scaler.transform([feature])[0]

    def get_label_name(self, index):
        """
        Maps model output index to a human-readable label using the label_map.
        :param index: Model output index.
        :return: Human-readable label for the index.
        """
        return self.label_map.get(index, f"Class_{index}")

    def save_scaler(self, path):
        """
        Saves the scaler used on the dataset in order to be used later on.
        :param index: Model output index.
        :return: Human-readable label for the index.
        """
        np.savez(path, mean=self.scaler.mean_, scale=self.scaler.scale_)

    def load_scaler(self, path):
        """
        Loads the scaler used on the dataset in order to be used for url images clustering.
        :param index: Model output index.
        :return: Human-readable label for the index.
        """
        data = np.load(path)
        self.scaler.mean_ = data['mean']
        self.scaler.scale_ = data['scale']