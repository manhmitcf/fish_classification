import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class FishDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0])  # Tên ảnh
        image = Image.open(img_name).convert("RGB")
        
        label = self.data.iloc[idx, 1:].values.astype(float)  # Lấy 6 giá trị cảm quan
        label = torch.tensor(label, dtype=torch.float32)  # Convert thành tensor
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Transform cho ảnh đầu vào
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
