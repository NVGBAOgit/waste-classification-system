import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import sys

# Import từ các file trong cùng thư mục
from efficientnet_finetune import build_efficientnet_finetune
from data_loader_finetune import train_loader, val_loader, class_weights

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 6
NUM_EPOCHS = 40
LEARNING_RATE_BACKBONE = 1e-5
LEARNING_RATE_CLASSIFIER = 1e-4

def train_finetune():
    print(f"=== Fine-tune EfficientNet-B0 ===")
    print(f"Device: {DEVICE}")
    
    model = build_efficientnet_finetune(num_classes=NUM_CLASSES, unfreeze_backbone=True, unfreeze_layers=0)
    model = model.to(DEVICE)
    
    # Đặt learning rate khác nhau cho backbone và classifier
    optimizer = Adam([
        {'params': model.features.parameters(), 'lr': LEARNING_RATE_BACKBONE},
        {'params': model.classifier.parameters(), 'lr': LEARNING_RATE_CLASSIFIER}
    ])
    
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    best_val_acc = 0
    
    for epoch in range(1, NUM_EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch}/{NUM_EPOCHS}")
        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        scheduler.step(avg_val_loss)
        
        # Lưu model tốt nhất
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "../saved_models/efficientnet_finetune_best.pth")
            print(f"--> Lưu model mới (Val Acc: {val_acc:.2f}%)")
    
    print(f"\n Hoàn thành! Best Val Accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    train_finetune()