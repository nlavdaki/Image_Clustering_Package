# image_downloader.py
import requests
import time
from PIL import Image
from io import BytesIO
from data_preprocessor import DataPreprocessor
from requests.exceptions import Timeout, ConnectionError

class ImageDownloader:
    def __init__(self, preprocessor: DataPreprocessor):
        self.preprocessor = preprocessor
        self.user_agents = [
            # List of User Agents to cycle through in case of connection issues
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:85.0) Gecko/20100101 Firefox/85.0",
        ]

    def download_image(self, url, retries=3):
        """
        Downloads an image from a given URL and preprocesses it.
        :param url: The URL of the image to download.
        :param retries: Number of retries for downloading the image.
        :return: A preprocessed PIL image or None if download fails.
        """
        for attempt in range(retries):
            try:
                headers = {
                    "User-Agent": self.user_agents[attempt % len(self.user_agents)],  # Cycle through user agents
                    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
                    "Accept-Encoding": "gzip, deflate, br",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Connection": "keep-alive"
                }

                print(f"Attempt {attempt + 1} to download image from {url}")
                response = requests.get(url, headers=headers, timeout=60, allow_redirects=True)
                response.raise_for_status()
                print("Image downloaded successfully.")

                # Open the image and convert it to RGB
                image = Image.open(BytesIO(response.content)).convert("RGB")

                # Preprocess the image
                print("Preprocessing the downloaded image.")
                preprocessed_image = self.preprocessor.preprocess_image(image)
                print("Image preprocessed successfully.")
                return preprocessed_image

            except (Timeout, ConnectionError) as e:
                print(f"Connection issue on attempt {attempt + 1}: {e}")
                if attempt < retries - 1:  # Wait before retrying if there are attempts left
                    backoff_time = 2 ** attempt  # Exponential backoff (2, 4, 8 seconds)
                    print(f"Waiting {backoff_time} seconds before retrying...")
                    time.sleep(backoff_time)

            except requests.RequestException as e:
                print(f"Attempt {attempt + 1} failed to download image from {url}: {e}")
                if attempt == retries - 1:  # Last attempt
                    print("All download attempts failed.")

            except Exception as e:
                print(f"Failed to preprocess image from {url}: {e}")
                return None

        return None
