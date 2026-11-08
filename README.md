Enhanced Visual Place Recognition using Local Global Feature Fusion
Overview
This repository contains the baseline implementation of a Visual Place Recognition (VPR) system using a pretrained DINOv2 model.

The baseline uses the GardensPointWalking dataset. It extracts global visual features from query and reference/database images, compares the features using cosine similarity, and retrieves the Top-5 most similar database images.
Dataset
- Dataset: GardensPointWalking
- Source: VPR-Bench GitHub repository
The dataset is organized into query images and reference/database images for the baseline experiment.
Baseline Workflow
1. Load the GardensPointWalking dataset.
2. Prepare the query and reference/database image paths.
3. Preprocess the images.
4. Load the pretrained DINOv2 model.
5. Extract global feature embeddings from the query and database images.
6. Normalize the feature vectors.
7. Calculate cosine similarity between query and database features.
8. Rank database images by similarity.
9. Retrieve and display the Top-5 matching images with their similarity scores.
Technologies
- Python
- PyTorch
- DINOv2
- NumPy
- OpenCV / Pillow
- Matplotlib
- Scikit-learn
Implementation Modules
- dataset.py – Loads the dataset and prepares query and database image paths.
- dinov2.py – Loads the pretrained DINOv2 model and extracts global image features.
- similarity.py – Calculates cosine similarity and ranks database images.
- main.py – Integrates the modules and displays the Top-5 retrieval results.
Project Structure
Enhanced-VPR/
├── datasets/
│   └── GardensPointWalking/
├── src/
│   ├── dataset.py
│   ├── dinov2.py
│   ├── similarity.py
│   └── main.py
├── results/
├── requirements.txt
└── README.md
Expected Output
For each query image, the system displays:
- The query image.
- The Top-5 most similar database images.
- The corresponding similarity scores.
Conclusion
This baseline uses pretrained DINOv2 for global feature extraction and cosine similarity for image retrieval on the GardensPointWalking dataset. It provides the initial implementation for the project Enhanced Visual Place Recognition using Local Global Feature Fusion.
