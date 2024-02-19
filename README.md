Car Detection with Faster R-CNN
===============================

This repository contains code for training and inference using Faster R-CNN (Region-based Convolutional Neural Network) for car detection. The code is organized into three main scripts:

1.  `train.py`: This script is responsible for training the Faster R-CNN model on a custom car detection dataset. It includes functions for loading the dataset, defining the model architecture, training the model, and saving the trained weights.

2.  `inference.py`: This script performs inference using the trained Faster R-CNN model. It loads a trained model's weights, selects a random image from the validation set, and visualizes the model's predictions, including bounding boxes and scores.

3.  `dataset.py`: This module defines a custom dataset class (`CarDetectionDataset`) for handling the car detection dataset. It includes data loading, augmentations, and collate functions.

Usage
-----

### 1\. Training the Model

To train the model, execute the following command:

```bash
python train.py
```
This will train the Faster R-CNN model for a specified number of epochs using the custom car detection dataset.

### 2\. Inference with the trained Model

For inference using the trained model, run the following command:
```bash
python inference.py`
```
This script loads the trained model, selects a random image from the validation set, and visualizes the model's predictions.

### 3\. Dataset Module

The `dataset.py` module contains the `CarDetectionDataset` class, which is used for loading and augmenting the car detection dataset. This class is utilized by both the training and inference scripts.

Dataset
-------

The car detection dataset used in this project can be downloaded from [here](https://www.kaggle.com/datasets/sshikamaru/car-object-detection/data). It includes images and corresponding bounding box annotations for car locations. The dataset is split into training and validation sets for model training and evaluation.

Configuration
-------------

Adjustments to hyperparameters, model architecture, or dataset paths can be made in the respective scripts (`train.py`, `inference.py`, and `dataset.py`).
