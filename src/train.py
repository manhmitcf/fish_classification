import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import FishDataset, transform
from model import FishClassifier
import os

# 📌 Thêm argparse để nhận tham số từ terminal
parser = argparse.ArgumentParser(description="Train Fish Classifier")
parser.add_argument("--epochs", type=int, default=20, help="Số epoch để train (default: 20)")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
args = parser.parse_args()

# Cấu hình từ argparse
TRAIN_CSV_PATH = "data/train.csv"
VAL_CSV_PATH = "data/val.csv"  # Thêm đường dẫn tới file validation CSV
IMG_DIR = "data/images/"
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.lr

# Load datasets
train_dataset = FishDataset(TRAIN_CSV_PATH, IMG_DIR, transform=transform)
val_dataset = FishDataset(VAL_CSV_PATH, IMG_DIR, transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FishClassifier().to(device)

# Loss và Optimizer
criterion = nn.MSELoss()  # Dùng MSELoss vì output là giá trị liên tục
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(EPOCHS):
    # Training phase
    model.train()
    train_running_loss = 0.0
    
    for images, labels in train_dataloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_running_loss += loss.item()
    
    avg_train_loss = train_running_loss / len(train_dataloader)
    
    # Validation phase
    model.eval()
    val_running_loss = 0.0
    
    with torch.no_grad():  # Tắt tính gradient trong validation
        for images, labels in val_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
    
    avg_val_loss = val_running_loss / len(val_dataloader)
    
    print(f"Epoch [{epoch+1}/{EPOCHS}], "
          f"Train Loss: {avg_train_loss:.4f}, "
          f"Val Loss: {avg_val_loss:.4f}")

# Lưu mô hình
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/resnet_model.pth")
print("Đã lưu mô hình!")