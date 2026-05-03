# Skin-Cancer-Detection
A deep learning project that classifies skin lesion images into multiple cancer categories using a Convolutional Neural Network (CNN). The project covers the complete machine learning pipeline — from raw data to a working prediction system.

## Overview
Skin cancer is one of the most common cancers globally, and early detection significantly improves survival rates. This project builds an image classification model that can identify the type of skin lesion from a dermoscopy image, automating a task that typically requires expert diagnosis.

## Dataset
Source: Dermoscopy image dataset with multiple skin lesion classes
Preprocessing includes class distribution analysis, augmentation to handle imbalanced classes, and train/validation/test splitting
Augmentation techniques applied to increase data diversity and reduce overfitting

## Requirements
tensorflow
keras
numpy
matplotlib
scikit-learn
opencv-python
Pillow

## Pipeline
Step 1 — Data Loading
Load images from directory, map class labels, and inspect class distribution.
Step 2 — Preprocessing
Resize images, normalize pixel values, and apply data augmentation (flips, rotations, zoom, brightness shifts) to balance underrepresented classes.
Step 3 — Model Architecture
Custom CNN built with TensorFlow/Keras:
Multiple convolutional + max-pooling layers
Batch normalization and dropout for regularization
Dense output layer with softmax activation for multi-class classification
Step 4 — Training
Optimizer: Adam
Loss: Categorical Crossentropy
Callbacks: EarlyStopping, ModelCheckpoint
Metrics tracked: accuracy and loss across epochs
Step 5 — Evaluation
Confusion matrix on test set
Sample prediction visualizations on validation and test images
Single-image local prediction support
