# Import required libraries 
import requests
import os
import tarfile

class FetchData:
    """
    A class to handle all data preparation tasks for the image classification project.

    This class is responsible for downloading the dataset, loading the data,
    and preprocessing it for use in the model. It provides methods to perform
    each of these steps individually or all at once.

    Attributes:
        data_dir (str): The directory where the dataset will be stored and processed.

    Methods:
        download_dataset: Downloads the dataset from a specified source.
        load_data: Loads the downloaded data into memory.
        preprocess_data: Applies necessary preprocessing to the loaded data.
        prepare: Executes all data preparation steps in sequence.
    """
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    file_path = 'dataset/Cifar.tar.gz'
    extract_path = 'dataset/cifar-10-batches-py'
    
    def __init__(self, url=None, file_path=None, extract_path=None) -> None:
        self.url = url or FetchData.url
        self.file_path = file_path or FetchData.file_path
        self.extract_path = extract_path or FetchData.extract_path
    
    def download_file(self):
        try:
            response = requests.get(self.url, stream=True)
            response.raise_for_status()
            # total_size = int(response.headers.get('content-length', 0))
            
            with open(self.file_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            
            print(f'File downloaded successfully. Size: {os.path.getsize(self.file_path) / (1024*1024):.2f} MB')
            return self.file_path
        except requests.exceptions.RequestException as err:
            print(f'An error occurred while downloading the file: {err}')
        except IOError as err:
            print(f'An error occurred while writing the file: {err}')
        return self.file_path if os.path.exists(self.file_path) else False
    def extract_file(self):
        try:
            with tarfile.open(self.file_path, 'r:gz') as tar:
                tar.extractall(path=self.extract_path)
            print(f'File extracted successfully to {self.extract_path}')
            return True
        except tarfile.TarError as err:
            print(f'An error occurred while extracting the file: {err}')
        return False