# GardensPointWalking Dataset Preparation

## 1. Dataset Information
Dataset Name: Gardens Point Walking  
Location: Queensland University of Technology, Brisbane, Australia  
Dataset Type: Visual Place Recognition (VPR)

## 2. Dataset Structure
The dataset contains three image sequences:

- day_left: Daytime traversal on the left side.
- day_right: Daytime traversal on the right side.
- night_right: Nighttime traversal on the right side.

## 3. Selected Experiment

Reference Database: day_left  
Query Dataset: night_right

Number of Database Images: 200  
Number of Query Images: 200  
Total Selected Images: 400

## 4. Image Preprocessing

- Images are loaded using the Pillow library.
- Images are converted into RGB format.
- Images are resized to 224 × 224 pixels.
- Original dataset images are preserved.

## 5. Technologies Used

- Python
- Pillow (PIL)
- NumPy
- Visual Studio Code

## 6. Generated Files

- dataset.py
- features/database_images.txt
- features/query_images.txt

## 7. Execution

Run the following command:

python dataset.py

The program identifies database and query image paths, tests image loading, and saves the prepared image lists.

## 8. Project Objective

To prepare image sequences for Visual Place Recognition using pretrained DINOv2 feature extraction and similarity-based image retrieval.