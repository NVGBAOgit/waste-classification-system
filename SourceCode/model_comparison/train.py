import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import train_loader, val_loader

# Import cả 3 model
from models.vgg16 import build_vgg16
from models.resnet50 import build_resnet50
from models.efficientnet import build_efficientnet

# Cấu hình
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 6
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

def run_training(model_name):
    print(f"ĐANG TRAIN MÔ HÌNH: {model_name.upper()}")
    print(f"\nĐang dùng: {DEVICE}")

    os.makedirs("saved_models", exist_ok=True)

    if model_name == "vgg16":
        model = build_vgg16(num_classes=NUM_CLASSES).to(DEVICE)
        save_path = "saved_models/vgg16_best.pth"
    elif model_name == "resnet50":
        model = build_resnet50(num_classes=NUM_CLASSES).to(DEVICE)
        save_path = "saved_models/resnet50_best.pth"
    elif model_name == "efficientnet":
        model = build_efficientnet(num_classes=NUM_CLASSES).to(DEVICE)
        save_path = "saved_models/efficientnet_best.pth"
    else:
        print("Tên mô hình sai!")
        return

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

    # Training loop
    def train_one_epoch(epoch):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        acc = 100. * correct / total
        avg_loss = total_loss / len(train_loader)
        return avg_loss, acc

    # Validation loop
    def validate():
        model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                total_loss += loss.item()

        acc = 100. * correct / total
        avg_loss = total_loss / len(val_loader)
        return avg_loss, acc

    # Chạy training
    best_val_acc = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(epoch)
        val_loss, val_acc = validate()
        scheduler.step()

        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"-> Model tốt nhất được lưu tại: {save_path} (Acc: {val_acc:.2f}%)")

    print(f"\nTraining {model_name.upper()} xong! Best Val Acc: {best_val_acc:.2f}%")

if __name__ == "__main__":
    # Để tên model muốn chạy
    run_training("efficientnet")