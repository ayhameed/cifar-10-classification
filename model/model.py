import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import torch.nn.functional as F  # Import the functional API
from torchsummary import summary
from sklearn.model_selection import train_test_split

# Get the path to the project root
project_root = os.path.abspath('..')

# Add project root to Python path
sys.path.append(project_root)

from image_preprocessing.imagePrep import *  # Import the PrepareData class

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(256 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 256 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
class ModelPrep():
    def __init__(self):
        pass
         
    def loss(self, model):
        # Create an instance of Model class
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss combines softmax and cross-entropy loss
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        
        return criterion, optimizer, scheduler
                
    def convert_to_tensors(self, X_train, X_test, y_train, y_test):
        # Convert data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).permute(0, 3, 1, 2)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)

        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).permute(0, 3, 1, 2)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        # Create PyTorch datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        # Create data loaders
        batch_size = 64
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_dataset, test_dataset, train_loader, test_loader
    
    def model_summary(self, model):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        return summary(model, input_size=(3, 32, 32))
    
    def evaluate_model(self, model, data_loader):
        model.eval()
        correct = 0
        total = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()  # Remove torch.argmax
        accuracy = 100 * correct / total
        return accuracy
    
    def train_model(self, model, criterion, optimizer, scheduler, train_loader, test_loader):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        num_epochs = 100

        best_accuracy = 0
        patience = 10
        no_improve = 0

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)  # Remove torch.argmax
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
            # Evaluation
            accuracy = self.evaluate_model(model, test_loader)
            print(f"Accuracy: {accuracy:.2f}%")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                no_improve = 0
                torch.save(model.state_dict(), 'saved_model/best_model.pth')
            else:
                no_improve += 1
            
            if no_improve == patience:
                print("Early stopping")
                break
            
            scheduler.step()