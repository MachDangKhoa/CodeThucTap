import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

# Định nghĩa hàm trích xuất đặc trưng
def extract_dino_features(image_path):
    """
    Trích xuất đặc trưng của ảnh sử dụng Vision Transformer (ViT) - DINOv2.
    """
    # Tải ảnh và chuyển sang định dạng RGB
    if isinstance(image_path, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB))
    elif isinstance(image_path, str):
        image = Image.open(image_path).convert("RGB")
    else:
        raise ValueError("Input image_path must be a file path (str) or a numpy.ndarray.")

    # Tiền xử lý ảnh trước khi đưa vào mô hình DINOv2 (ViT)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize ảnh cho phù hợp với DINOv2
        transforms.ToTensor(),  # Chuyển đổi ảnh thành tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization cho ViT
    ])
    image = transform(image).unsqueeze(0)  # Thêm chiều batch

    # Khởi tạo mô hình ViT (sử dụng pretrained ViT từ torchvision)
    model = models.vision_transformer.vit_b_16(pretrained=True)
    model.eval()  # Chuyển mô hình về chế độ đánh giá (evaluation)

    # Dự đoán đặc trưng của ảnh
    with torch.no_grad():
        features = model(image)

    # Chuyển tensor thành numpy array để sử dụng
    features = features.squeeze().cpu().numpy()
    return features