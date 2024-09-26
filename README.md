# CIFAR-10 Image Classification
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
   ```sh
   git clone https://github.com/ayhameed/cifar-10-classification
   cd cifar-10-classification
   ```
2. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

## Dataset
The CIFAR-10 dataset is automatically downloaded and extracted using the `download_and_extract_dataset.py` script. To download and extract the dataset, run:
   ```sh
   python3 download_and_extract_dataset.py
   ```

## Usage

### Training the Model
1. Ensure you have downloaded and extracted the CIFAR-10 dataset as described in the Dataset section.
2. Run the `main.py` script to preprocess the data, build the model, and train it:
   ```sh
   python3 main.py
   ```
   This will preprocess the CIFAR-10 dataset, build a Convolutional Neural Network (CNN) model, and train it. The best model will be saved in the `saved_model` directory.

## File Structure
```
cifar-10-classification/
│
├── dataset/
│   └── cifar-10-batches-py/  # Directory where CIFAR-10 data batches are stored
│
├── data/
|   └── getDataset.py         # Script to download and extract the CIFAR-10 dataset
│   └── prepareData.py        # Script to preparing the CIFAR-10 dataset
│
├── image_preprocessing/
│   └── imagePrep.py          # Script for image preprocessing tasks
│
├── visualize/
│   └── visualize.py          # Script for visualizing the CIFAR-10 dataset
│
├── model/
│   └── model.py              # Script containing the CNN model and training logic
│
├── saved_model/              # Directory where the trained model will be saved
│
├── download_and_extract_dataset.py  # Script to download and extract the CIFAR-10 dataset
├── main.py                   # Main script to run the entire pipeline
├── requirements.txt          # List of required packages
└── README.md                 # This README file
```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any changes.

## License
This project is licensed under the MIT License.

### Memory Forensics
Step 1: Install gdb (mac)
```sh
brew install gdb
```

Step 2: Code Signing GDB
macOS requires GDB to be code-signed to run properly. Follow these steps to sign GDB:
1. Open the Keychain Access application.
2. In the menu bar, go to Keychain Access > Certificate Assistant > Create a Certificate.
3. Name the certificate (e.g., gdb-cert), set Identity Type to Self Signed Root, and Certificate Type to Code Signing.
4. Click Create and then close the dialog.
5. Find your new certificate in the Keychain Access, right-click it, and select Get Info.
6. Expand the Trust section and set Code Signing to Always Trust.
7.Close the dialog and enter your password to save changes.

Step 4: Sign GDB
1. Run the following command in the Terminal to sign GDB:
```Bash
   sudo codesign -s gdb-cert $(which gdb)
```

Step 5: Run GDB
1. Run the following command in the Terminal to start GDB:
```Bash
   gdb
```
