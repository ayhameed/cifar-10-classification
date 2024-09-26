'''
This script downloads the dataset and extracts it to the  dataset directory 
into a file named cifar-20-batches.py

PS: please delete the tar.gz file on completion
'''
# Import the Prepare Data Class
from data.getDataset import FetchData

# Create a new instance of the prepare data class 
data = FetchData()
# Download dataset
dataset = data.download_file()
    
# extract dataset
dataset = data.extract_file()