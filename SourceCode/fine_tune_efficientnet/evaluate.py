import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import os

from efficientnet_finetune import build_efficientnet_finetune
from data_loader import test_loader, NUM_CLASSES, CLASS_NAMES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model(model_path):
    print(f"Đánh giá model fine-tune")
    print(f"Device: {DEVICE}")

    model = build_efficientnet_finetune(num_classes=NUM_CLASSES, unfreeze_backbone=True)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = np.mean(np.array(all_preds) == np.array(all_labels)) * 100
    print(f"\nTest Accuracy: {acc:.2f}%")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - EfficientNet Fine-tune (Acc: {acc:.2f}%)')
    plt.tight_layout()

    save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'outputs', 'confusion_matrix_finetune_final.png'))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"\nĐã lưu confusion matrix tại: {save_path}")

    return cm, acc

if __name__ == "__main__":
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'saved_models', 'efficientnet_finetune_best.pth'))
    if not os.path.exists(model_path):
        print(f"Không tìm thấy model tại {model_path}. Hãy chạy train.py trước.")
    else:
        evaluate_model(model_path)