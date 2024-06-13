import torch
import numpy as np
import pickle
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from itertools import cycle
from imblearn.over_sampling import SMOTE
import os
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Define transformations and dataset
data_dir = '/content/drive/MyDrive/Classes_SMOTE2'
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

# Load the saved model parameters
with open("global_parameters.pkl", "rb") as f:
    global_parameters = pickle.load(f)

# Define model and load parameters
model = models.resnet101(pretrained=False)
num_classes = len(dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

params_dict = zip(model.state_dict().keys(), global_parameters)
state_dict = {k: torch.tensor(v) for k, v in params_dict}
model.load_state_dict(state_dict, strict=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Split dataset and create DataLoader
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
train_size = int(0.8 * len(balanced_dataset))
test_size = len(balanced_dataset) - train_size
_, test_dataset = random_split(balanced_dataset, [train_size, test_size])

test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

# Evaluate model
all_labels = []
all_preds = []
all_probs = []
with torch.no_grad():
    for inputs, labels in test_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        probs = torch.softmax(outputs, dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

# Classification report
print(classification_report(all_labels, all_preds, target_names=dataset.classes))

# Confusion matrix
conf_matrix = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=dataset.classes, yticklabels=dataset.classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# ROC curve
all_labels_bin = label_binarize(all_labels, classes=range(num_classes))
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(all_labels_bin[:, i], np.array(all_probs)[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plotting the ROC curves
plt.figure(figsize=(14, 10))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple', 'brown', 'pink', 'gray', 'black'])
for i, color in zip(range(num_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve of class {i} (area = {roc_auc[i]:0.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for Each Class')
plt.legend(loc="lower right")
plt.show()
