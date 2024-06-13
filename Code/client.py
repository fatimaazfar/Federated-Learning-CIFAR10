import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, Subset
from imblearn.over_sampling import SMOTE
from typing import Dict, Tuple
import sys
import os
from PIL import Image
import random
from collections import defaultdict

data_dir = r'D:\Users\Nadeem\Desktop\BSDS\FYP\Data\Classes_SMOTE2-20240430T230050Z-001\Skin_Data'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CustomImageFolder(datasets.ImageFolder):
    def find_classes(self, directory):
        classes = []
        class_to_idx = {}
        for folder_name in sorted(os.listdir(directory)):
            if folder_name.startswith('ROI_'):
                class_name = folder_name.split('ROI_')[-1]
            else:
                class_name = folder_name
            classes.append(class_name)
            class_to_idx[folder_name] = len(classes) - 1
        return classes, class_to_idx

dataset = CustomImageFolder(root=data_dir, transform=transform)

def get_features_and_labels(dataset):
    features = []
    labels = []
    for img, label in dataset:
        features.append(img.numpy())
        labels.append(label)
    return np.array(features), np.array(labels)

features, labels = get_features_and_labels(dataset)

n_samples, n_channels, height, width = features.shape
features_reshaped = features.reshape(n_samples, -1)

smote = SMOTE()
features_resampled, labels_resampled = smote.fit_resample(features_reshaped, labels)

features_resampled = features_resampled.reshape(-1, n_channels, height, width)

class BalancedDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.features[idx]
        label = self.labels[idx]
        img = torch.tensor(img)
        if self.transform:
            img = self.transform(img)
        return img, label

balanced_dataset = BalancedDataset(features_resampled, labels_resampled)

# Ensure each client gets only 2 images per class
client_id = int(sys.argv[1])
class_indices = defaultdict(list)
for idx, (_, label) in enumerate(balanced_dataset):
    class_indices[label].append(idx)

client_indices = []
for class_idx in class_indices:
    selected_indices = class_indices[class_idx][client_id*2:(client_id+1)*2]
    client_indices.extend(selected_indices)

# Create a Subset for the client dataset
client_dataset = Subset(balanced_dataset, client_indices)

train_size = int(0.8 * len(client_dataset))
test_size = len(client_dataset) - train_size
train_dataset, test_dataset = random_split(client_dataset, [train_size, test_size])

batch_size = 1  # Update to 1 since each client has only a few images
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

num_classes = len(dataset.classes)

def create_model(num_classes):
    model = models.resnet101(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# Flower client
class CifarClient(fl.client.NumPyClient):
    def __init__(self, model, train_dataloader, test_dataloader):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        running_loss = 0.0
        for inputs, labels in self.train_dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        return self.get_parameters(config), len(self.train_dataloader.dataset), {"loss": running_loss / len(self.train_dataloader.dataset)}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.test_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        return running_loss / len(self.test_dataloader.dataset), len(self.test_dataloader.dataset), {"accuracy": accuracy}

# Initialize the model
model = create_model(num_classes)

# Start the client
def start_client():
    client = CifarClient(model, train_dataloader, test_dataloader)
    fl.client.start_client(server_address="localhost:8082", client=client)

if __name__ == "__main__":
    start_client()
