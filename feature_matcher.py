import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import os

# ========== CONFIG ==========
FEATURES_PKL = "D:/LuanVanTotNghiep/NhanDienThongTinTranh/features/features.pkl"
SIM_THRESHOLD = 0.80

# Cấu hình API Google Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyA-Lymc-kEm_p00YMcQEbAlvgO79ZO-fgQ")
genai.configure(api_key=GOOGLE_API_KEY)

# ========== Load Features ==========
with open(FEATURES_PKL, "rb") as f:
    features_data = pickle.load(f)
    features_array = features_data["features"]
    filenames = features_data["filenames"]
    styles = features_data["styles"]


# ========== Cosine Matching ==========
def find_best_match_cosine(input_vector, threshold=SIM_THRESHOLD):
    scores = cosine_similarity(input_vector, features_array)[0]
    best_idx = np.argmax(scores)
    best_score = scores[best_idx]

    if best_score >= threshold:
        return {
            "file_name": filenames[best_idx],
            "style": styles[best_idx],
            "similarity": round(best_score, 4),
            "description": generate_description(filenames[best_idx], styles[best_idx])
        }
    return None


# ========== Generate Description with Gemini ==========
def generate_description(file_name, style):
    prompt = (f"Hãy viết mô tả ngắn gọn và tóm tắt các ý chính về bức tranh '{file_name}' theo phong cách '{style}'. "
              "Tập trung vào các yếu tố chính như phong cách nghệ thuật và chủ đề của bức tranh, tránh dài dòng.")

    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        description = response.text if response and response.text else "Không có mô tả."

        # Cắt chuỗi mô tả bắt đầu từ từ khóa "Mô tả"
        start_index = description.find("Mô tả:")
        if start_index != -1:
            return description[start_index:]
        return description  # Nếu không tìm thấy "Mô tả", trả về toàn bộ mô tả

    except Exception as e:
        print(f"Lỗi khi gọi API Gemini: {e}")
        return "Không thể tạo mô tả."

