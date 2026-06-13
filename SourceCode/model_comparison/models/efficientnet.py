import torch
import torch.nn as nn
from torchvision import models

def build_efficientnet(num_classes=6, freeze=True):
    # Load EfficientNet-B0 pretrained từ ImageNet
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    
    # Freeze các lớp đầu (không train lại toàn bộ mạng để tiết kiệm thời gian)
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
            
    # Thay lớp cuối từ 1000 class → 6 class
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    return model

if __name__ == "__main__":
    model = build_efficientnet(num_classes=6)
    print(model)
    print(f"\nSố lớp được train: {sum(p.requires_grad for p in model.parameters())}")