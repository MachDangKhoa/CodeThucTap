import os
import numpy as np
import pandas as pd
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, jsonify
from selenium_search import search_google_and_extract_info
from feature_matcher import find_best_match_cosine
from selenium_search import process_image_for_search

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
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # remove FC layer
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

@app.route("/predict", methods=["GET","POST"])
def predict():
    if request.method == "GET":
        return '''
                <h2>Upload ảnh để nhận diện tranh</h2>
                <form method="POST" enctype="multipart/form-data">
                    <input type="file" name="image" accept="image/*" required>
                    <input type="submit" value="Upload">
                </form>
            '''
        # Ghi nội dung form vào file debug_page.html
        with open("D:/LuanVanTotNghiep/NhanDienThongTinTranh/templates/debug_page.html", "w", encoding="utf-8") as f:
            f.write(form_html)
        return form_html

    if "image" not in request.files:
        return jsonify({"error": "Không có ảnh được cung cấp"}), 400

    image_file = request.files["image"]
    image = Image.open(image_file).convert("RGB")
    temp_image_path = "temp.jpg"
    image.save(temp_image_path)

    try:
        input_vector = extract_feature_vector(temp_image_path)
        best_match = find_best_match_cosine(input_vector)

        if best_match:
            photographer, painting_title = parse_painting_info(best_match["file_name"])
            author = get_artist_from_class(best_match["file_name"])

            info = {
                "painting_title": str(painting_title),
                "photographer": str(photographer),
                "style": str(best_match["style"]),
                "artist": str(author),
                "similarity": float(best_match["similarity"]),
                "matched_file": str(best_match["file_name"])
            }

            return jsonify({"source": "Dataset Cosine", "info": info})

        else:
            # Nếu không tìm thấy trong Dataset → fallback sang Google Search
            google_info = search_google_and_extract_info(temp_image_path)

            if "error" in google_info:
                return jsonify({"error": "Không tìm thấy tranh trong dataset hoặc từ Google Search"})

            def convert_to_serializable(obj):
                if isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(v) for v in obj]
                elif isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                else:
                    return obj
            return jsonify(convert_to_serializable(google_info))
    finally:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)


# ========== Run App ==========
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

