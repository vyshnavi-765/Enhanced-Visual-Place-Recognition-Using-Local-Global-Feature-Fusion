
import os

root = r"C:\Users\ganga\Downloads\GardensPointWalking"

print("Dataset exists:", os.path.exists(root))

total = 0

for current, folders, files in os.walk(root):
    images = [
        f for f in files
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if images:
        print("\nFolder:", current)
        print("Image count:", len(images))
        print("Samples:", images[:3])
        total += len(images)

print("\nTotal images:", total)
