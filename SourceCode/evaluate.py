import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import test_loader, train_dataset
from models.vgg16 import build_vgg16

# Cấu hình
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 6
CLASS_NAMES = train_dataset.classes

# Load model đã train
model = build_vgg16(num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load("saved_models/vgg16_best.pth", map_location=DEVICE))
model.eval()

# Đánh giá trên tập test
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Tính accuracy
accuracy = 100. * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
print(f"Test Accuracy: {accuracy:.2f}%")

# In classification report (precision, recall, F1-score)
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

# Vẽ confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Confusion Matrix - VGG16')
plt.ylabel('Nhãn thật')
plt.xlabel('Nhãn dự đoán')
plt.tight_layout()
plt.savefig('outputs/confusion_matrix_vgg16.png')
plt.show()
print("Confusion matrix đã lưu vào outputs/confusion_matrix_vgg16.png")