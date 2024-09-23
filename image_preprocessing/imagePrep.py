import numpy as np

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