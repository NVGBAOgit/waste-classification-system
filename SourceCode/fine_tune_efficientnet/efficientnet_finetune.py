import torch
import torch.nn as nn
from torchvision import models

def build_efficientnet_finetune(num_classes=10, unfreeze_backbone=True, unfreeze_layers=0):
    # Load pretrained EfficientNet-B0 từ ImageNet
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    
    # Freeze toàn bộ tham số ban đầu
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze backbone theo tùy chọn
    if unfreeze_backbone:
        if unfreeze_layers > 0:
            # Unfreeze N lớp cuối của features
            children = list(model.features.children())
            
            # Chốt chặn an toàn: Tránh lỗi out of index
            if unfreeze_layers > len(children):
                unfreeze_layers = len(children)
                
            for i in range(-unfreeze_layers, 0):
                for param in children[i].parameters():
                    param.requires_grad = True
        else:
            # Unfreeze toàn bộ backbone (chỉ features)
            for param in model.features.parameters():
                param.requires_grad = True
    
    # Thay classifier để khớp với số class mới (mặc định là 10)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    return model