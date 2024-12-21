# data_preprocessor.py
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

class DataPreprocessor:
    def __init__(self, target_size=(224, 224), use_clahe=True, mask_bubbles=False, use_edge_detection=False):
        self.target_size = target_size
        self.use_clahe = use_clahe
        self.mask_bubbles = mask_bubbles
        self.use_edge_detection = use_edge_detection

    def resize_with_aspect_ratio(self, img):
        """Resizes image while maintaining aspect ratio and adds padding to meet target size."""
        h, w = img.shape[:2]
        scale = min(self.target_size[0] / h, self.target_size[1] / w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Padding to reach target size
        delta_w, delta_h = self.target_size[1] - new_w, self.target_size[0] - new_h
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    def apply_clahe(self, img_array):
        """Applies CLAHE (contrast limited adptive histogram equalization) to enhance image contrast."""
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))

        ## Investigate if this is good for your dataset cause sometimes it isnt
        final_img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        # Optional normalization to scale pixel values between 0-255
        final_img = cv2.normalize(final_img, None, 0, 255, cv2.NORM_MINMAX)

        print("After CLAHE (adjusted):", final_img.shape, final_img[0, 0, :])  # Sample pixel values

        return final_img

    # This is for this specific dataset due to bubble peculiarities https://colorizer.org/, https://hyperskill.org/learn/step/13179
    def remove_colored_bubbles(self, img):
        """Removes specific colored bubbles from the image based on HSV range filtering."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_color = np.array([0, 35, 35])
        upper_color = np.array([360, 40, 45])
        mask = cv2.inRange(hsv, lower_color, upper_color)
        mask_inv = cv2.bitwise_not(mask)
        background_only = cv2.bitwise_and(img, img, mask=mask_inv)

        return cv2.GaussianBlur(background_only, (3, 3), 0)

    # This is mostly an optimization in combination with bubble removal (above)
    def apply_edge_detection(self, img):
        """Applies Canny edge detection to the image."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, threshold1=50, threshold2=150)

        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Function that applies all the above
    def preprocess_image(self, img_input):
        """
        Preprocesses an image from a file path or PIL Image object with CLAHE, edge detection, etc.
        :param img_input: A file path (str) or a PIL.Image object.
        :return: A preprocessed PIL image ready for model input.
        """
        if isinstance(img_input, str):
            # Load image from file path
            img = cv2.imread(img_input)
            if img is None:
                raise ValueError(f"Image at path '{img_input}' could not be read.")
        elif isinstance(img_input, Image.Image):
            # Convert PIL image to OpenCV format
            img = cv2.cvtColor(np.array(img_input), cv2.COLOR_RGB2BGR)
        else:
            raise TypeError("Expected a file path or PIL Image object for img_input.")

        # Resize, apply CLAHE, edge detection, etc.
        img = self.resize_with_aspect_ratio(img)
        if self.mask_bubbles:
            img = self.remove_colored_bubbles(img)
        if self.use_clahe:
            img = self.apply_clahe(img)
        if self.use_edge_detection:
            img = self.apply_edge_detection(img)

        # Gaussian Blur and convert to RGB format for PIL compatibility
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return Image.fromarray(img_rgb)