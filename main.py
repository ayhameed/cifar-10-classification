
'''
Prepare the Data
'''

# Import the Prepare Data Class
from data.getDataset import getData

# Create a new instance of the prepare data class 
data = getData()
# Download dataset
dataset = data.download_file()
    
# extract dataset
dataset = data.extract_file()