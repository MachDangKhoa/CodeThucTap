import os
import numpy as np
import pandas as pd
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from feature_matcher import find_best_match_cosine
from flask_cors import CORS


# ========== CONFIG ==========
DATASET_PATH = "D:/LuanVanTotNghiep/NhanDienThongTinTranh/wikiart"
VAL_CSV = "D:/LuanVanTotNghiep/NhanDienThongTinTranh/data/artist_val.csv"
CLASS_TXT = "D:/LuanVanTotNghiep/NhanDienThongTinTranh/data/artist_class.txt"

# ========== Load Mappings ==========
class_to_artist = {}
with open(CLASS_TXT, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(" ", 1)
        if len(parts) == 2:
            class_id = int(parts[0])
            artist_name = parts[1].replace("_", " ")
            class_to_artist[class_id] = artist_name

# Đọc file val.csv
val_df = pd.read_csv(VAL_CSV, header=None, names=["filename", "class_id"])

# Tạo mapping filename (chỉ tên file, bỏ path) → class_id
filename_to_class = {
    os.path.basename(filename).strip().lower(): int(class_id)
    for filename, class_id in zip(val_df["filename"], val_df["class_id"])
}

# ========== Load Model ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = models.resnet50(pretrained=True)
resnet.fc = torch.nn.Identity()
resnet.to(device).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ========== Feature Extraction ==========
def extract_feature_vector(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = resnet(img_tensor)
    return features.squeeze().cpu().numpy().reshape(1, -1)

# ========== Parse Info from File Name ==========
def parse_painting_info(file_name):
    """
    Trích xuất thông tin từ tên file dạng: aaron-siskind_acolman-1-1955.jpg
    → photographer: aaron-siskind
    → painting_title: acolman-1-1955
    """
    try:
        filename = os.path.splitext(os.path.basename(file_name))[0]
        if "_" in filename:
            parts = filename.split("_")
            photographer = parts[0]
            painting_title = "_".join(parts[1:])
            return photographer, painting_title
        else:
            return "Không rõ", filename
    except Exception:
        return "Không rõ", "Không rõ"

def get_artist_from_class(file_name):
    file_name = os.path.basename(file_name).strip().lower()
    class_id = filename_to_class.get(file_name, None)
    if class_id is not None:
        return class_to_artist.get(class_id, "Không rõ")
    return "Không rõ"

# ========== Flask API ==========
app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return '''
            <h2>Upload ảnh để nhận diện tranh</h2>
            <form method="POST" enctype="multipart/form-data">
                <input type="file" name="image" accept="image/*" required>
                <input type="submit" value="Upload">
            </form>
        '''

    if "image" not in request.files:
        return jsonify({"error": "Không có ảnh được cung cấp"}), 400

    image_file = request.files["image"]

    if image_file.filename == "":
        return jsonify({"error": "Tên file không hợp lệ"}), 400
    temp_image_path = "temp.jpg"
    try:
        # Lưu ảnh tạm thời
        image = Image.open(image_file).convert("RGB")
        image.save(temp_image_path)

        # 1️⃣ Tìm trong dataset trước
        input_vector = extract_feature_vector(temp_image_path)
        best_match = find_best_match_cosine(input_vector)

        if best_match:
            photographer, painting_title = parse_painting_info(best_match["file_name"])
            author = get_artist_from_class(best_match["file_name"])

            info = {
                "painting_title": painting_title,
                "artist": author,
                "style": best_match["style"],
                "photographer": photographer,
                "similarity": float(best_match["similarity"]),
                "matched_file": best_match["file_name"],
                "description": best_match.get("description", "Không có mô tả.")
            }

            return jsonify({"source": "Dataset Cosine", "info": info})

        # ❌ Nếu không tìm thấy gì cả → Trả lỗi cuối cùng
        return jsonify({"error": "Không tìm thấy tranh trong dataset, Google Search hoặc xử lý ảnh tải xuống."})

    finally:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)


# ========== Run App ==========
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

