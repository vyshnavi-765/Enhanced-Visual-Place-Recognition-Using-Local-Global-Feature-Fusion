
import os
import csv
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
FEATURE_DIR = os.path.join(ROOT, "features")
RESULT_DIR = os.path.join(ROOT, "results")

os.makedirs(RESULT_DIR, exist_ok=True)

print("Loading extracted features...")

database_features = np.load(
    os.path.join(FEATURE_DIR, "database_features.npy")
)

query_features = np.load(
    os.path.join(FEATURE_DIR, "query_features.npy")
)

with open(os.path.join(FEATURE_DIR, "database_images.txt")) as f:
    database_images = [line.strip() for line in f if line.strip()]

with open(os.path.join(FEATURE_DIR, "query_images.txt")) as f:
    query_images = [line.strip() for line in f if line.strip()]

if len(database_images) != len(database_features):
    raise ValueError("Database image and feature counts do not match")

if len(query_images) != len(query_features):
    raise ValueError("Query image and feature counts do not match")

print("Database features:", database_features.shape)
print("Query features:", query_features.shape)

def cosine_similarity(query, database):
    query_norm = query / np.maximum(
        np.linalg.norm(query, axis=1, keepdims=True), 1e-12
    )

    database_norm = database / np.maximum(
        np.linalg.norm(database, axis=1, keepdims=True), 1e-12
    )

    return np.dot(query_norm, database_norm.T)


print("\nCalculating cosine similarity...")

similarity_matrix = cosine_similarity(
    query_features,
    database_features
)

print("Similarity matrix:", similarity_matrix.shape)

output_file = os.path.join(RESULT_DIR, "top5_matches.csv")

with open(output_file, "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "Query Image",
        "Rank",
        "Database Image",
        "Similarity Score"
    ])

    for i, scores in enumerate(similarity_matrix):

        top_indices = np.argsort(scores)[::-1][:5]

        for rank, index in enumerate(top_indices, start=1):

            writer.writerow([
                query_images[i],
                rank,
                database_images[index],
                round(float(scores[index]), 6)
            ])

print("\nTop-5 matches for first query image:")

top_indices = np.argsort(similarity_matrix[0])[::-1][:5]

for rank, index in enumerate(top_indices, start=1):

    print("\nRank:", rank)
    print("Database image:", database_images[index])
    print("Similarity score:", round(
        float(similarity_matrix[0][index]), 6
    ))

print("\nImage retrieval completed!")
print("Total query images:", len(query_images))
print("Top-5 results saved:", output_file)
