# CIFAR-10 Image Classification Project

This project focuses on downloading, processing, and using the CIFAR-10 dataset for image classification tasks.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Getting Started](#getting-started)
3. [Dataset](#dataset)
4. [Usage](#usage)
5. [File Structure](#file-structure)
6. [Contributing](#contributing)
7. [License](#license)

## Project Overview

This project aims to work with the CIFAR-10 dataset, a widely used benchmark in machine learning for image classification. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes.

## Getting Started

### Prerequisites
- Python 3.x
- pip (Python package installer)

### Installation
1. Clone this repository:
   ```
   git clone https://github.com/ayhameed/cifar-10-classification.git
   cd cifar-10-classification
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Dataset

The CIFAR-10 dataset is automatically downloaded and extracted using the `download_dataset.py` script.

To download and extractthe dataset, run:
```
python download_dataset.py
```

## Usage

To train a model on the CIFAR-10 dataset, use the following command:
```
run the image_classification_v2 jupyter notebook
```

This will train a simple convolutional neural network (CNN) model using the CIFAR-10 dataset.

## File Structure

- `download_dataset.py`: Script to download and extract the CIFAR-10 dataset
- `train_model.py`: Script to train a CNN model on the CIFAR-10 dataset 
- `model.py`: Script to define the CNN model architecture
- `train.py`: Script to train the CNN model
- `predict.py`: Script to predict the class of an image
- `utils.py`: Utility functions for data loading and visualization      
- `requirements.txt`: List of required packages
- `README.md`: This file    


