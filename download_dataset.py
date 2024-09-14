'''
This script Downloads and Extracts the CIFAR-10 Dataset
How to use this script:
        
    1. Ensure you have the required libraries installed:
        pip3 install requests
        
    2. Run the script:
        python3 download_dataset.py
        
    3. The script will download the CIFAR-10 dataset and extract it.
        - The downloaded file will be saved as 'Cifar.tar.gz'
        - The extracted data will be in the 'cifar-10-data' directory
        
    4. After successful execution, the dataset will be ready for use in your projects.
        
    Note: Make sure you have sufficient disk space (about 170 MB) and a stable internet connection.
'''

import requests
import os
import tarfile

dataset_url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
file_path = 'Cifar.tar.gz'
extract_path = 'cifar-10-batches-py'

def download_file():
    try:
        response = requests.get(dataset_url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        
        print(f'File downloaded successfully. Size: {os.path.getsize(file_path) / (1024*1024):.2f} MB')
        return True
    except requests.exceptions.RequestException as err:
        print(f'An error occurred while downloading the file: {err}')
    except IOError as err:
        print(f'An error occurred while writing the file: {err}')
    return False

def extract_file():
    try:
        with tarfile.open(file_path, 'r:gz') as tar:
            tar.extractall(path=extract_path)
        print(f'File extracted successfully to {extract_path}')
        return True
    except tarfile.TarError as err:
        print(f'An error occurred while extracting the file: {err}')
    return False

if __name__ == "__main__":
    if download_file():
        if extract_file():
            print("Dataset is ready for use.")
        else:
            print("Failed to extract the dataset.")
    else:
        print("Failed to download the dataset.")
        
        
