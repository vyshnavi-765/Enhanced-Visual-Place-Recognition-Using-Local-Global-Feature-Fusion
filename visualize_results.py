
import os
import csv
import matplotlib.pyplot as plt
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))

csv_file = os.path.join(ROOT, "results", "top5_matches.csv")

with open(csv_file, "r") as file:
    rows = list(csv.DictReader(file))

first_query = rows[0]["Query Image"]

matches = [
    row for row in rows
    if row["Query Image"] == first_query
]

fig, axes = plt.subplots(1, 6, figsize=(20, 5))

query = Image.open(first_query).convert("RGB")
axes[0].imshow(query)
axes[0].set_title("Query Image\nNight Right")
axes[0].axis("off")

for i, match in enumerate(matches):
    image_path = match["Database Image"]
    score = float(match["Similarity Score"])

    image = Image.open(image_path).convert("RGB")

    axes[i + 1].imshow(image)
    axes[i + 1].set_title(
        f"Top {i + 1}\nScore: {score:.4f}"
    )
    axes[i + 1].axis("off")

plt.tight_layout()

output = os.path.join(
    ROOT, "results", "top5_visualization.png"
)

plt.savefig(output, dpi=150)
plt.show()

print("Visualization saved:", output)
