import torch
from torch.utils.data import DataLoader
from dataset import FishDataset, transform
from model import FishClassifier
import numpy as np

# Load mô hình
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FishClassifier().to(device)
model.load_state_dict(torch.load("models/resnet_model.pth"))
model.eval()

# Load test dataset
CSV_PATH = "data/labels.csv"
IMG_DIR = "data/images/"
dataset = FishDataset(CSV_PATH, IMG_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# Đánh giá
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        
        all_preds.append(outputs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

# Convert về numpy
all_preds = np.vstack(all_preds)
all_labels = np.vstack(all_labels)

# Tính RMSE cho từng feature
rmse = np.sqrt(np.mean((all_preds - all_labels) ** 2, axis=0))
print(f"RMSE từng điểm cảm quan: {rmse}")
