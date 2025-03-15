import torch
from torchvision import transforms
from PIL import Image
from model import FishClassifier

# Load mô hình
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FishClassifier().to(device)
model.load_state_dict(torch.load("models/resnet_model.pth"))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
    
    return output.cpu().numpy().flatten()

# Test
img_path = "data/images/sample.jpg"
print(f"Điểm cảm quan dự đoán: {predict(img_path)}")
