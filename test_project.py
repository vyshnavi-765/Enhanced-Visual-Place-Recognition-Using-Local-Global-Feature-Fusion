import os
import numpy as np
import pandas as pd

print("MODULE 1: DATASET")

print("MODULE 2: FEATURES")

db = np.load("features/database_features.npy")
query = np.load("features/query_features.npy")

print("Database shape:", db.shape)
print("Query shape:", query.shape)

assert db.ndim == 2
assert query.ndim == 2
assert db.shape[1] == query.shape[1]
assert np.isfinite(db).all()
assert np.isfinite(query).all()

print("Feature extraction verified!")

print("\nMODULE 3: RETRIEVAL")

df = pd.read_csv("results/top5_matches.csv")
print(df.head())

assert len(df) >= 5

print("Top-5 results verified!")

assert os.path.exists("results/top5_visualization.png")

print("Visualization exists!")

print("\nALL AVAILABLE OUTPUT CHECKS PASSED!")