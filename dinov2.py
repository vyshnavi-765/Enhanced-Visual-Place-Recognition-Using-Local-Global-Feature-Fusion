
import os
import torch
import numpy as np

from PIL import Image
from transformers import AutoImageProcessor, AutoModel

ROOT = os.path.dirname(os.path.abspath(__file__))
FEATURE_DIR = os.path.join(ROOT, "features")

MODEL_NAME = "facebook/dinov2-base"

device = torch.device("cpu")

print("Loading DINOv2 model...")

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

model = model.to(device)
model.eval()

print("DINOv2 model loaded successfully!")
print("Device:", device)


def read_image_list(filename):
    path = os.path.join(FEATURE_DIR, filename)

    with open(path, "r") as file:
        return [line.strip() for line in file if line.strip()]


def extract_features(image_paths):
    features = []

    for i, image_path in enumerate(image_paths):

        image = Image.open(image_path).convert("RGB")

        inputs = processor(images=image, return_tensors="pt")

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = model(**inputs)

            feature = outputs.last_hidden_state[:, 0, :]

            feature = torch.nn.functional.normalize(
                feature, p=2, dim=1
            )

        features.append(feature.squeeze(0).cpu().numpy())

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(image_paths)} images")

    return np.array(features, dtype=np.float32)


database_images = read_image_list("database_images.txt")
query_images = read_image_list("query_images.txt")

print("\nExtracting database features...")
database_features = extract_features(database_images)

print("\nExtracting query features...")
query_features = extract_features(query_images)

np.save(
    os.path.join(FEATURE_DIR, "database_features.npy"),
    database_features
)

np.save(
    os.path.join(FEATURE_DIR, "query_features.npy"),
    query_features
)

print("\nFeature extraction completed!")

print("Database features:", database_features.shape)
print("Query features:", query_features.shape)

print("\nFeature files saved successfully!")
