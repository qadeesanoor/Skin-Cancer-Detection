# Skin Cancer Detection

A deep learning project that classifies skin lesion images into multiple skin cancer categories using a **Convolutional Neural Network (CNN)**.

The project covers the complete image classification pipeline, including data preprocessing, image augmentation, class balancing, model training, evaluation, and single-image prediction.

## Overview

Skin lesion classification is an important computer vision application in the medical imaging domain.

This project uses dermoscopy images to train a CNN that learns visual patterns associated with different skin lesion classes.

The complete workflow includes:

* Loading and organizing image data
* Analyzing class distribution
* Image preprocessing
* Data augmentation
* Train, validation, and test splitting
* CNN model development
* Model training
* Performance evaluation
* Confusion matrix visualization
* Single-image prediction

## Project Workflow

```text
Skin Lesion Images
        |
        v
Data Loading
        |
        v
Class Distribution Analysis
        |
        v
Image Preprocessing
        |
        v
Data Augmentation
        |
        v
Train / Validation / Test Split
        |
        v
CNN Model
        |
        v
Model Training
        |
        v
Model Evaluation
        |
        v
Single Image Prediction
```

## Dataset

The project uses a dermoscopy image dataset containing multiple skin lesion classes.

The dataset is processed to:

* Inspect class distribution
* Identify class imbalance
* Resize images
* Normalize pixel values
* Increase training data through augmentation
* Divide the dataset into training, validation, and testing sets

Data augmentation is used to increase image diversity and help reduce overfitting.

## Image Preprocessing

Before being passed to the CNN, images undergo preprocessing.

The pipeline includes:

* Image resizing
* Pixel normalization
* Random horizontal and vertical transformations
* Rotation
* Zoom
* Brightness adjustments
* Other augmentation techniques

These transformations help the model learn more robust visual patterns.

## Model Architecture

The project uses a custom **Convolutional Neural Network (CNN)** implemented with TensorFlow/Keras.

The architecture includes:

* Convolutional layers
* Max-pooling layers
* Batch normalization
* Dropout layers
* Fully connected dense layers
* Softmax output layer

The final softmax layer produces probabilities for the different skin lesion classes.

## Training

The model is trained using:

```text
Optimizer: Adam
Loss Function: Categorical Crossentropy
```

The training process also uses callbacks such as:

* EarlyStopping
* ModelCheckpoint

Training and validation performance are monitored using:

* Accuracy
* Loss

## Evaluation

The trained model is evaluated using the test dataset.

The project includes:

### Confusion Matrix

A confusion matrix is generated to analyze the model's classification performance across the different classes.

It helps identify:

* Correct predictions
* Incorrect predictions
* Classes that are commonly confused with each other

### Prediction Visualization

Sample validation and test images can also be passed through the model to visualize the predicted classes.

## Single Image Prediction

The project supports prediction on an individual skin lesion image.

The general prediction workflow is:

```text
Input Image
     |
     v
Resize Image
     |
     v
Normalize Image
     |
     v
CNN Model
     |
     v
Class Probabilities
     |
     v
Predicted Class
```

The model returns the predicted skin lesion category based on the learned image features.

## Technologies and Libraries

The project is implemented using Python and the following libraries:

* **TensorFlow** — Deep learning framework
* **Keras** — Neural network development
* **NumPy** — Numerical operations
* **Matplotlib** — Visualization
* **Scikit-learn** — Machine learning utilities and evaluation
* **OpenCV** — Image processing
* **Pillow** — Image handling

## Requirements

Create a `requirements.txt` file containing:

```text
tensorflow
keras
numpy
matplotlib
scikit-learn
opencv-python
Pillow
```

These are the dependencies currently listed by the project.

## Installation

Clone the repository:

```bash
git clone https://github.com/qadeesanoor/Skin-Cancer-Detection.git
```

Move into the project directory:

```bash
cd Skin-Cancer-Detection
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

## Running the Project

### Using Jupyter Notebook

Open:

```text
skin_cancer_detection.ipynb
```

Run the notebook cells sequentially to perform:

1. Dataset loading
2. Data exploration
3. Image preprocessing
4. Data augmentation
5. Model creation
6. Model training
7. Model evaluation
8. Confusion matrix generation
9. Image prediction

### Using Python

The project also includes:

```text
skin_cancer_detection.py
```

Run it using:

```bash
python skin_cancer_detection.py
```

## Project Structure

```text
Skin-Cancer-Detection/
│
├── Images/
│
├── frontend/
│
├── app.py
├── skin_cancer_detection.ipynb
├── skin_cancer_detection.py
├── Report of Notebook.pdf
├── project video.mp4
└── README.md
```

## Flask Application

The repository also contains an `app.py` file for the prediction application.

This provides a way to use the trained model as part of an application rather than only running predictions inside the notebook.

The project therefore contains both:

* Model development through Jupyter Notebook/Python
* Application-based prediction through Flask

## Key Features

* Deep learning-based skin lesion classification
* Custom CNN architecture
* Image preprocessing
* Data augmentation
* Class imbalance handling
* Train/validation/test data split
* Batch normalization
* Dropout regularization
* Early stopping
* Model checkpointing
* Confusion matrix evaluation
* Single-image prediction
* Flask-based application component

## Limitations

This project is intended for **educational and research purposes**.

Skin lesion classification from images is a complex medical task, and predictions from this model should not be considered a medical diagnosis. A qualified healthcare professional should be consulted for actual medical assessment.

Model performance can also be affected by:

* Image quality
* Dataset size
* Class imbalance
* Lighting and imaging conditions
* Differences between training and real-world images
* Similar visual characteristics between different lesion classes

## Future Improvements

Possible improvements include:

* Using transfer learning with architectures such as ResNet, EfficientNet, or MobileNet
* Increasing the size and diversity of the dataset
* Applying advanced class-balancing techniques
* Hyperparameter tuning
* Cross-validation
* Adding precision, recall, and F1-score
* Adding ROC-AUC analysis
* Improving the prediction interface
* Adding confidence scores to predictions
* Deploying the model as a web application
* Adding explainability techniques such as Grad-CAM

## Conclusion

This project demonstrates how deep learning and computer vision can be applied to skin lesion image classification.

Using a custom CNN, the project covers the complete workflow from image preprocessing and augmentation to model training, evaluation, and single-image prediction.

It provides a practical implementation of a medical image classification pipeline using **Python and TensorFlow/Keras**.
