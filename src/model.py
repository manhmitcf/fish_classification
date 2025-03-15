import torch
import torch.nn as nn
import torchvision.models as models

class FishClassifier(nn.Module):
    def __init__(self):
        super(FishClassifier, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 6)  # Output 6 giá trị cảm quan
        
    def forward(self, x):
        return self.resnet(x)
