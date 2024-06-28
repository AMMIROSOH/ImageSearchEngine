import json
import requests
from PIL import Image
from io import BytesIO
import torch
import clip
import time

class ClipAI:
    def __init__(self, modelName="ViT-B/32", device=None):
        """
        Initialize the ImageEncoder with the specified model and device.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(modelName, device=self.device)

    def downloadImage(self, url, productId, max_retries=10, delay=2):
        """
        Download an image from a given URL.
        """
        for attempt in range(max_retries):
            try:
                response = requests.get(url)
                response.raise_for_status()  # Raise an exception for HTTP errors
                # with open(f'E:\\Datasets\\ImageSearch\\{productId}.jpg', 'wb') as f:
                #     f.write(response.content)
                return Image.open(BytesIO(response.content))
            except Exception as e:
                print(f"Attempt {attempt + 1} failed with error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(delay)
                else:
                    raise FileNotFoundError


    def encodeImage(self, image):
        """
        Encode an image using the CLIP model.
        """
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            imageFeatures = self.model.encode_image(image)
        return imageFeatures / imageFeatures.norm(dim=-1, keepdim=True)

    def encodeText(self, text):
        """
        Encode a given text using the CLIP model.
        """
        with torch.no_grad():
            textTokens = clip.tokenize([text]).to(self.device)
            textFeatures = self.model.encode_text(textTokens)
        return textFeatures / textFeatures.norm(dim=-1, keepdim=True)

    def saveFeatures(self, filePath):
        """
        Save the encoded features to a JSON file.
        """
        with open(filePath, 'w', encoding='utf-8') as f:
            json.dump(self.features, f)

    def processProduct(self, product):
        """
        Process a product, downloading and encoding it image.
        """
        imageUrl = product['images'][0]
        image = self.downloadImage(imageUrl, product['id'])
        features = self.encodeImage(image)
        return features.cpu().numpy().tolist()
