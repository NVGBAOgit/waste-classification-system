import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os

DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))  # Đường dẫn tới thư mục data gốc

# Augmentation mạnh hơn cho train
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),
])

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load dataset
train_dataset = datasets.ImageFolder(DATASET_PATH, transform=train_transforms)
val_test_dataset = datasets.ImageFolder(DATASET_PATH, transform=val_test_transforms)

# Chia train/val/test
indices = list(range(len(train_dataset)))
labels = [train_dataset.targets[i] for i in indices]
train_indices, val_test_indices = train_test_split(indices, test_size=0.3, stratify=labels, random_state=42)
val_indices, test_indices = train_test_split(val_test_indices, test_size=0.5, stratify=[labels[i] for i in val_test_indices], random_state=42)

train_set = Subset(train_dataset, train_indices)
val_set = Subset(val_test_dataset, val_indices)
test_set = Subset(val_test_dataset, test_indices)

# Class weights từ train set
train_labels = [train_dataset.targets[i] for i in train_indices]
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weights = torch.tensor(class_weights, dtype=torch.float32)

# DataLoader
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

print(f"Train samples: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
print(f"Class weights: {class_weights}")
