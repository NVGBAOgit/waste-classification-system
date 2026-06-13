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

# Import 3 model
from models.vgg16 import build_vgg16
from models.resnet50 import build_resnet50
from models.efficientnet import build_efficientnet

# Cấu hình
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 6
CLASS_NAMES = train_dataset.classes

def evaluate_model(model_name):
    os.makedirs("outputs", exist_ok=True)
    # Chọn model 
    if model_name == "vgg16":
        model = build_vgg16(num_classes=NUM_CLASSES).to(DEVICE)
        model_path = "saved_models/vgg16_best.pth"
        title = 'Confusion Matrix - VGG16'
        save_path = 'outputs/confusion_matrix_vgg16.png'
    elif model_name == "resnet50":
        model = build_resnet50(num_classes=NUM_CLASSES).to(DEVICE)
        model_path = "saved_models/resnet50_best.pth"
        title = 'Confusion Matrix - ResNet50'
        save_path = 'outputs/confusion_matrix_resnet50.png'
    elif model_name == "efficientnet":
        model = build_efficientnet(num_classes=NUM_CLASSES).to(DEVICE)
        model_path = "saved_models/efficientnet_best.pth"
        title = 'Confusion Matrix - Efficientnet'
        save_path = 'outputs/confusion_matrix_efficientnet.png'
    else:
        print("Tên model không hợp lệ!")
        return

    # Load model đã train
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
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
    plt.title(title) # Đổi tiêu đề
    plt.ylabel('Nhãn thật')
    plt.xlabel('Nhãn dự đoán')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Confusion matrix đã lưu vào {save_path}")

if __name__ == "__main__":
    evaluate_model("efficientnet")