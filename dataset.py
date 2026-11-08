
import os
from PIL import Image

ROOT = r"C:\Users\ganga\Downloads\GardensPointWalking"

DATABASE_FOLDER = os.path.join(ROOT, "day_left")
QUERY_FOLDER = os.path.join(ROOT, "night_right")

def get_images(folder):
    images = []

    for file in sorted(os.listdir(folder)):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            images.append(os.path.join(folder, file))

    return images

database_images = get_images(DATABASE_FOLDER)
query_images = get_images(QUERY_FOLDER)

print("Database images:", len(database_images))
print("Query images:", len(query_images))

print("\nDatabase samples:")
for path in database_images[:5]:
    print(path)

print("\nQuery samples:")
for path in query_images[:5]:
    print(path)

def load_image(path):
    image = Image.open(path).convert("RGB")
    image = image.resize((224, 224))
    return image

if database_images and query_images:
    image = load_image(database_images[0])

    print("\nImage loading successful!")
    print("Image size:", image.size)
    print("Image mode:", image.mode)
else:
    print("Error: No images found.")
FEATURE_FOLDER = "features"

os.makedirs(FEATURE_FOLDER, exist_ok=True)

with open(os.path.join(FEATURE_FOLDER, "database_images.txt"), "w") as f:
    for path in database_images:
        f.write(path + "\n")

with open(os.path.join(FEATURE_FOLDER, "query_images.txt"), "w") as f:
    for path in query_images:
        f.write(path + "\n")

print("\nImage lists saved successfully!")