import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import os
import mysql.connector
import openai


# ========== CONFIG ==========
SIM_THRESHOLD = 0.80

# Cấu hình kết nối MySQL
db_config = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': '',
    'database': 'paint_recogn'
}

# Kết nối đến MySQL
conn = mysql.connector.connect(**db_config)

# Tạo con trỏ để thực hiện các truy vấn
cursor = conn.cursor()

# Truy vấn lấy model_path từ bảng paintings_models
cursor.execute("SELECT model_path FROM painting_models WHERE is_active = 1")

# Lấy tất cả kết quả truy vấn
models_data = cursor.fetchall()

# Kiểm tra và xử lý dữ liệu
if models_data:
    for model in models_data:
        model_path = model[0]  # model_path là cột đầu tiên trong kết quả truy vấn
        FEATURES_PKL = model_path
else:
    print("Không tìm thấy model_path trong cơ sở dữ liệu.")

# Truy vấn lấy thông tin cấu hình từ bảng api_configs
cursor.execute("SELECT api_key, model_name, llm_name FROM api_configs LIMIT 1")

# Lấy kết quả truy vấn
config = cursor.fetchone()

#Cấu hình API
if config is not None:
    api_key = config[0]      # api_key từ cơ sở dữ liệu
    model_name = config[1]   # model_name từ cơ sở dữ liệu
    llm_name = config[2]     # llm_name từ cơ sở dữ liệu

    print(f"LLM Name: {llm_name}")
    print(f"API Key: {api_key}")
    print(f"Model Name: {model_name}")

    # Cấu hình API Google Gemini hoặc ChatGPT
    if llm_name == 'Gemini':
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
    elif llm_name == 'ChatGPT':
        openai.api_key = api_key
        model = model_name
        pass
else:
    print("Không tìm thấy cấu hình trong cơ sở dữ liệu.")

# Đóng con trỏ và kết nối
cursor.close()
conn.close()

# # Cấu hình API Google Gemini
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyA-Lymc-kEm_p00YMcQEbAlvgO79ZO-fgQ")
# genai.configure(api_key=GOOGLE_API_KEY)
# model = genai.GenerativeModel("GPT-4o")

# ========== Load Features ==========
FEATURES_PKL = FEATURES_PKL .replace("\\", "/")
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
        if llm_name == 'Gemini':
            response = model.generate_content(prompt)
            description = response.text if response and response.text else "Không có mô tả."
        elif llm_name == 'ChatGPT':
            # Cấu hình và gọi OpenAI API cho ChatGPT
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150
            )
            description = response.choices[0].message.content.strip() if response and response.choices else "Không có mô tả."

        # Cắt chuỗi mô tả bắt đầu từ từ khóa "Mô tả"
        start_index = description.find("Mô tả:")
        if start_index != -1:
            return description[start_index:]
        return description  # Nếu không tìm thấy "Mô tả", trả về toàn bộ mô tả

    except Exception as e:
        print(f"Lỗi khi gọi API: {e}")
        return "Không thể tạo mô tả."

