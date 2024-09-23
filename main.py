'''
    Set Path
'''
import sys, os
# Get the path to the project root
project_root = os.path.abspath('..')
# Add project root to Python path
sys.path.append(project_root)

'''
    Import Classes
'''
from data.prepareData import *
from image_preprocessing.imagePrep import *
from visualize.visualize import *
from model.model import *

'''
Load Data Set
'''
'''
Create Instance of prepareData Class
'''
dataset = getData # Create new instance of getData Class

meta_data = dataset.load_file(file='./dataset/cifar-10-batches-py/cifar-10-batches-py/batches.meta') # Load Meta_data
all_data = [] # to hold all data
b_1 = dataset.load_file(file='./dataset/cifar-10-batches-py/cifar-10-batches-py/data_batch_1') # for batch one 
all_data.append(b_1) # Append to list
b_2 = dataset.load_file(file='./dataset/cifar-10-batches-py/cifar-10-batches-py/data_batch_2') # for batch 2 
all_data.append(b_2) # Append to list
b_3 = dataset.load_file(file='./dataset/cifar-10-batches-py/cifar-10-batches-py/data_batch_3') # for batch 3
all_data.append(b_3) # Append to list
b_4 = dataset.load_file(file='./dataset/cifar-10-batches-py/cifar-10-batches-py/data_batch_4') # for batch 4
all_data.append(b_4) # Append to list
b_5 = dataset.load_file(file='./dataset/cifar-10-batches-py/cifar-10-batches-py/data_batch_5') # for batch 5
all_data.append(b_5) # Append to list

'''
Visualize some data
'''
visual = Visualize # create instance of visualize class

## Plot and visualize the image from the first data batch in all_data array
images = all_data[0][b'data']
labels = all_data[0][b'labels']
img_prep = ImagePreprocessing # Create an instance of the Image preprocessing class
r_images = img_prep.reshape_image(images) #reshape image
print(f'Image Shape {np.shape(r_images)}, Label Shape {np.shape(labels)}')
# remove comment to plot 
# visual.plot_first_10(images=r_images, labels=labels, meta_data=meta_data)
merged_data, merged_labels = dataset.merge_datasets(all_data=all_data) ### Merge all data batches
rm_images = img_prep.reshape_image(merged_data) # Reshape the merged images
### Plot firstt 10 from merged data remove comment to plot 
# visual.plot_first_10(images=rm_images, labels=merged_labels, meta_data=meta_data)

'''
Image preprocessing
'''
img_prep = ImagePreprocessing(rm_images, merged_labels) # Create instance of imagePrep class

X_train, X_test, y_train, y_test = img_prep.split_data(rm_images, merged_labels) # Split data
X_train = img_prep.normalize(X_train) # Normalize Images in X_train
X_test = img_prep.normalize(X_test) # Normalize Images in X_tests

'''
Prepare tensors, Build Model
'''
model = CNNModel() # Create instance of model class 

modelprep = ModelPrep() # Create instance of model prep class 

criterion, optimizer, scheduler = modelprep.loss(model) # calling criterion, optimzer and scheduler from this poorly named function
train_dataset, test_dataset, train_loader, test_loader = modelprep.convert_to_tensors(X_train, X_test, y_train, y_test)
print(modelprep.model_summary(model)) # Get model summary
#print(modelprep.train_model(model, criterion, optimizer, scheduler, train_loader, test_loader))# evaluate model

'''
Train Model
'''
print(modelprep.train_model(model, criterion, optimizer, scheduler, train_loader, test_loader=test_loader)) ## train model