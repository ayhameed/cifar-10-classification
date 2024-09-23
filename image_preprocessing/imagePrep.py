import numpy as np
from sklearn.model_selection import train_test_split

class ImagePreprocessing:
    """
    A class to preprocess images for machine learning tasks.

    Attributes:
    ----------
    images : np.ndarray
        A numpy array of images.
    labels : np.ndarray
        A numpy array of labels corresponding to the images.

    Methods:
    -------
    reshape_image(images):
        Reshapes images to 32x32x3 format.
    normalize(images):
        Normalizes images to have pixel values between 0 and 1.
    split_data(images, labels):
        Splits the dataset into train and test sets.
    """

    def __init__(self, images, labels):
        """
        Constructs all the necessary attributes for the ImagePreprocessing object.

        Parameters:
        ----------
        images : np.ndarray
            A numpy array of images.
        labels : np.ndarray
            A numpy array of labels corresponding to the images.
        """
        self.images = images
        self.labels = labels

    @staticmethod
    def reshape_image(images):
        """
        Reshapes images to 32x32x3 format.

        Parameters:
        ----------
        images : np.ndarray
            A numpy array of images to be reshaped.

        Returns:
        -------
        np.ndarray
            A numpy array of reshaped images.
        """
        reshaped_images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        return reshaped_images

    @staticmethod
    def normalize(images):
        """
        Normalizes images to have pixel values between 0 and 1.

        Parameters:
        ----------
        images : np.ndarray
            A numpy array of images to be normalized.

        Returns:
        -------
        np.ndarray
            A numpy array of normalized images.
        """
        normalized_images = images.astype('float32') / 255.0
        return normalized_images

    @staticmethod
    def split_data(images, labels):
        """
        Splits the dataset into train and test sets.

        Parameters:
        ----------
        images : np.ndarray
            A numpy array of images.
        labels : np.ndarray
            A numpy array of labels corresponding to the images.

        Returns:
        -------
        tuple
            A tuple containing the train and test splits: (X_train, X_test, y_train, y_test).
        """
        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
        return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)