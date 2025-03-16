import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights  # Import weights mới

class FishClassifier(nn.Module):
    def __init__(self):
        super(FishClassifier, self).__init__()
        # Sử dụng weights thay vì pretrained
        self.resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 6)  # Output 6 classes
        
    def forward(self, x):
        return self.resnet(x)
