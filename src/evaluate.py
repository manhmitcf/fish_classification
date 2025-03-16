import torch
from torch.utils.data import DataLoader
from dataset import FishDataset, transform
from model import FishClassifier
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Load mô hình
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FishClassifier().to(device)
model.load_state_dict(torch.load("models/resnet_model.pth"))
model.eval()

# Load test dataset
CSV_PATH = "data/val.csv"
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

# Làm tròn tất cả các giá trị dự đoán
rounded_preds = np.round(all_preds)

# Đảm bảo giá trị nằm trong khoảng [1, 9]
rounded_preds = np.clip(rounded_preds, 1, 9)

# Chuyển đổi về integer (vì classification metrics yêu cầu số nguyên)
rounded_preds = rounded_preds.astype(int)
all_labels = all_labels.astype(int)

# Tính accuracy và F1-score cho từng cột (feature)
for i in range(6):
    acc = accuracy_score(all_labels[:, i], rounded_preds[:, i])
    f1 = f1_score(all_labels[:, i], rounded_preds[:, i], average="macro")
    print(f"Feature {i+1} - Accuracy: {acc:.4f}, F1-score: {f1:.4f}")

# In RMSE
rmse = np.sqrt(np.mean((all_preds - all_labels) ** 2, axis=0))
print(f"RMSE từng điểm cảm quan: {rmse}")
