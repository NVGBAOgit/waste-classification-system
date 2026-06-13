import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os

DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')) 

CLASS_NAMES = ['battery', 'cardboard', 'clothes', 'glass', 'metal', 
               'organic', 'paper', 'plastic', 'shoes', 'trash']
NUM_CLASSES = len(CLASS_NAMES)

train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load dataset
full_dataset = datasets.ImageFolder(DATASET_PATH, transform=train_transforms)
val_test_dataset = datasets.ImageFolder(DATASET_PATH, transform=val_test_transforms)

# Chia train/val/test
indices = list(range(len(full_dataset)))
labels = [full_dataset.targets[i] for i in indices]
train_idx, valtest_idx = train_test_split(indices, test_size=0.3, stratify=labels, random_state=42)
val_idx, test_idx = train_test_split(valtest_idx, test_size=0.5, stratify=[labels[i] for i in valtest_idx], random_state=42)

train_set = Subset(full_dataset, train_idx)
val_set = Subset(val_test_dataset, val_idx)
test_set = Subset(val_test_dataset, test_idx)

# Class weights
train_labels = [labels[i] for i in train_idx]
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weights = torch.tensor(class_weights, dtype=torch.float32)

# DataLoader
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

print(f"Classes: {CLASS_NAMES}")
print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
print(f"Class weights: {class_weights}")