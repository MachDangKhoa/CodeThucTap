import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# ========== CONFIG ==========
FEATURES_PKL = "D:/LuanVanTotNghiep/NhanDienThongTinTranh/features/features.pkl"
SIM_THRESHOLD = 0.80

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
            "similarity": round(best_score, 4)
        }
    return None
