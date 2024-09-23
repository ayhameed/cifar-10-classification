# Import required libraries 
import pickle
from sklearn.model_selection import train_test_split
import numpy as np

class getData:
    """
    A class to handle all data preparation tasks for the image classification project.

    This class is responsible for downloading the dataset, loading the data,
    and preprocessing it for use in the model. It provides methods to perform
    each of these steps individually or all at once.

    Attributes:
        file (str): The file to be unpickled.
    Methods:
        load_file(file: str) -> dict:
            Uses Pickle to unpickle the data.
    """
    all_data = [] # Hold an array of datasets and labels
    
    def __init__(self, file: str) -> None:
        """
        Initializes the getData class with the specified file.

        Args:
            file (str): The file to be unpickled.
        """
        self.file = file
        
    def load_file(file: str) -> dict:
        """
        Uses Pickle to unpickle the data.

        Args:
            file (str): The file to be unpickled.

        Returns:
            dict: A dictionary containing the unpickled data.
        """
        with open(file, 'rb') as fo:
            data = pickle.load(fo, encoding='bytes')
        return data
    
    def merge_datasets(all_data):
        '''
            Merges an array of datasets 
            Params : all_data :list - holding all data and labels
            Returns : merged_data, merged_labels
        '''
        data = []
        labels = []

        # Loop through each batch in the data_batches list and append the data and labels
        for batch in all_data:
            data.append(batch[b'data'])
            labels.append(batch[b'labels'])
        # Concatenate all the data (image arrays) and labels into single numpy arrays
        merged_data = np.concatenate(data, axis=0)
        merged_labels = np.concatenate(labels, axis=0)
        
        return merged_data, merged_labels