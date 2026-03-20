import torch
import torch.nn as nn
from torch.optim import Adam # Thuật toán cập nhật trọng số
from torch.optim.lr_scheduler import StepLR # StepLR tự động giảm learning rate sau một số epoch nhất định
import sys
import os

# Thêm đường dẫn để import các module khác
sys.path.append(os.path.dirname(os.path.abspath(__file__))) # Định vị vị trí tìm file import

from data_loader import train_loader, val_loader, train_dataset
from models.vgg16 import build_vgg16

# Cấu hình
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 6
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

print(f"Đang dùng: {DEVICE}")

# Khởi tạo model
model = build_vgg16(num_classes=NUM_CLASSES).to(DEVICE)

# Loss function và optimizer
criterion = nn.CrossEntropyLoss() # Tính độ sai của model
optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE) # Cập nhật tham số
scheduler = StepLR(optimizer, step_size=3, gamma=0.1) # Cứ 3 epoch giảm learning rate 10%

# Training loop
def train_one_epoch(epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0 # Tổng số ảnh đã xử lý

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # Forward pass
        outputs = model(images) # Đưa ảnh vào model
        loss = criterion(outputs, labels) # So sánh kết quả của model với nhãn đúng

        # Backward pass
        optimizer.zero_grad() # Xóa gradient từ batch trước
        loss.backward() # Tính gradient của batch hiện tại
        optimizer.step() # Cập nhật trọng số

        # Tính accuracy
        _, predicted = outputs.max(1) # Lấy index class có xác suất cao nhất
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
    model.eval() # Chế độ đánh giá
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad(): # Tắt tính gradient
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

    print(f"Epoch {epoch}/{NUM_EPOCHS}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    # Lưu model tốt nhất
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "saved_models/vgg16_best.pth") # Lưu toàn bộ trọng số vào file vgg16_best.pth
        print(f"Model tốt nhất được lưu! Val Acc: {val_acc:.2f}%")

print(f"\nTraining xong! Best Val Acc: {best_val_acc:.2f}%")