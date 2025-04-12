import cv2
import numpy as np

def crop_painting(image_path, output_path):
    # Đọc ảnh
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Dùng adaptive threshold để tách nền và tranh
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Loại bỏ nhiễu bằng phép toán Morphology
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Tìm contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Không tìm thấy bức tranh trong ảnh!")
        return None

    # Chọn contour lớn nhất (giả sử là tranh)
    largest_contour = max(contours, key=cv2.contourArea)

    # Tính hình chữ nhật bao quanh tranh
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_image = image[y:y+h, x:x+w]

    # Lưu ảnh kết quả
    cv2.imwrite(output_path, cropped_image)
    return output_path