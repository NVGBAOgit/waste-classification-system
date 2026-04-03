import torch
import torch.nn as nn
from torchvision import models

def build_efficientnet_finetune(num_classes=6, unfreeze_backbone=True, unfreeze_layers=0):

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    
    for param in model.parameters():
        param.requires_grad = False
    
    if unfreeze_backbone:
        if unfreeze_layers > 0:
            # Unfreeze N lớp cuối của features
            children = list(model.features.children())
            for i in range(-unfreeze_layers, 0):
                for param in children[i].parameters():
                    param.requires_grad = True
        else:
            # Unfreeze toàn bộ backbone
            for param in model.parameters():
                param.requires_grad = True
    
    # Thay classifier 
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    return model