import torch
import torch.nn as nn
from torchvision import models

def build_resnet50(num_classes=6, freeze=True):
    # Load ResNet50 pretrained từ ImageNet
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # Freeze các lớp đầu (không train lại)
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
    
    # Thay lớp cuối từ 1000 class → 6 class
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model

if __name__ == "__main__":
    model = build_resnet50(num_classes=6)
    print(model)
    print(f"\nSố lớp được train: {sum(p.requires_grad for p in model.parameters())}")