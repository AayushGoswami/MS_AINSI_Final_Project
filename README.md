# Fruit Classification Using Deep Learning

## Overview

This project provides a complete workflow for classifying fresh and rotten fruits (apples, bananas, oranges) using deep learning and transfer learning with PyTorch. The solution is designed for automated quality control in fruit supply chains, helping companies quickly and accurately distinguish between fresh and rotten produce using image data.

> **Note:** The current implementation uses static images for fruit classification. In the future, this project can be scaled up to identify and classify fruits from a live video feed using OpenCV, enabling real-time quality control on conveyor belts or in industrial settings.

## Problem Statement

A leading fruit supply company requires an automated system to classify fresh and rotten fruits in real-time from conveyor belt images. Manual sorting is slow and inconsistent, leading to customer complaints. This project leverages computer vision to automate fruit classification, improving speed and accuracy in quality control.

## Project Structure

```
fruit_classification.ipynb   # Main Jupyter notebook with the full workflow
utils.py                    # Utility functions for training and validation
requirements.txt            # Python dependencies
images/
    logo.png                # Project logo
    fruits.png              # Example fruit images
LICENSE                     # MIT License
README.md                   # Project documentation
```

## Features

- **Transfer Learning:** Uses a pre-trained VGG16 model from torchvision, customized for 6 fruit classes (fresh/rotten apples, bananas, oranges).
- **Data Augmentation:** Applies random rotations, crops, and flips to improve model generalization.
- **Custom Dataset Loader:** Efficiently loads and preprocesses images from labeled directories.
- **Training & Fine-Tuning:** Supports both feature extraction and full fine-tuning of the VGG16 model.
- **Evaluation:** Reports accuracy and loss on validation data.
- **Model Saving:** Saves the trained model for future inference.

## Workflow

1. **Import Libraries:** Loads PyTorch, torchvision, and other dependencies.
2. **Download Dataset:** Uses `kagglehub` to fetch the "fruits-fresh-and-rotten-for-classification" dataset.
3. **Data Visualization:** Displays sample images for inspection.
4. **Model Initialization:** Loads VGG16 with pre-trained weights, freezes base layers, and customizes the classifier.
5. **Data Augmentation:** Defines transformations for training images.
6. **Custom Dataset:** Implements a PyTorch `Dataset` to load and preprocess images.
7. **DataLoader Setup:** Batches and shuffles data for training and validation.
8. **Training:** Trains the model and validates after each epoch.
9. **Fine-Tuning:** Unfreezes all layers and continues training with a lower learning rate.
10. **Evaluation:** Assesses final model performance.
11. **Model Saving:** Saves the fine-tuned model to disk.

## Usage

1. **Install Dependencies**

   ```sh
   pip install -r requirements.txt
   ```

2. **Run the Notebook**

   Open `fruit_classification.ipynb` in Jupyter or VS Code and execute the cells sequentially.

3. **Dataset**

   The notebook automatically downloads the dataset using KaggleHub. Ensure you have access to Kaggle datasets.

4. **Training and Evaluation**

   The notebook will train the model, validate its performance, and save the trained weights.

## Utility Functions

Key training and validation utilities are implemented in `utils.py`:

- `train`: Trains the model for one epoch.
- `validate`: Evaluates the model on validation data.
- `get_batch_accuracy`: Computes batch accuracy.

## Requirements

See `requirements.txt` for all dependencies:

- torch
- torchvision
- kagglehub
- Pillow
- glob2
- os
- ipykernel
- jupyter

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgements

- Fruit dataset: [Kaggle - Fruits Fresh and Rotten for Classification](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification)
- PyTorch and torchvision for deep learning frameworks.

---
*Developed by Aayush Goswami, 2025*