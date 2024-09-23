import matplotlib.pyplot as plt
import numpy as np
import os, sys

# Get the path to the project root
project_root = os.path.abspath('..')

# Add project root to Python path
sys.path.append(project_root)

from data.prepareData import *  # Import the PrepareData class

class Visualize(getData):
    """
    A class to visualize images from a dataset.

    This class inherits from the PrepareData class and is responsible for displaying images and their labels from the dataset.

    Attributes:
        images (list): A list of images to be plotted.
        labels (list): A list of labels corresponding to the images.

    Methods:
        plot_first_10(load_metadata, batches_metadata):
            Plots the first 10 images with their labels.
    """
    
    def __init__(self) -> None:
        """
        Initializes the Visualize class with images and labels.

        Args:
            images (list): A list of images to be plotted.
            labels (list): A list of labels corresponding to the images.
        """
        super().__init__()  # Initialize the base class
       
        
    def plot_first_10(images, labels, meta_data):
        """
        Plots the first 10 images with their labels.

        This method loads metadata from a specified file, retrieves label names, and plots the first 10 images with their corresponding labels.

        Args:
            images (list): A list of images to be plotted.
            labels (list): A list of labels corresponding to the images.
            metaDataFile (dict): The dict holding the metadata file.

        Returns:
            None
        """
        label_names = meta_data[b'label_names']
        # Plot the first 10 images
        plt.figure(figsize=(10, 2))
        for i in range(10):
            plt.subplot(1, 10, i + 1)
            plt.imshow(images[i])
            plt.title(label_names[labels[i]].decode('utf-8'))  # decode bytes to string
            plt.axis('off')  # Turn off axis
        plt.show()