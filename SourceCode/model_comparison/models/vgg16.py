import torch
import torch.nn as nn
from torchvision import models

def build_vgg16(num_classes=6, freeze=True):
    # Load VGG16 pretrained từ ImageNet
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    
    # Freeze các lớp đầu (không train lại)
    if freeze:
        for param in model.features.parameters():
            param.requires_grad = False # Tắt điều chỉnh trọng số những phần đã đóng băng
    
    # Thay lớp cuối từ 1000 class → 6 class
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    
    return model

if __name__ == "__main__":
    model = build_vgg16(num_classes=6)
    print(model)
    print(f"\nSố lớp được train: {sum(p.requires_grad for p in model.parameters())}")