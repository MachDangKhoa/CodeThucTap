import base64
import os
import time
import uuid
import json
from skimage.transform import resize
import requests
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import undetected_chromedriver as uc
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from selenium.webdriver.common.action_chains import ActionChains
from extract_dino_features import extract_dino_features
from detect_image import crop_painting
import google.generativeai as genai
import re
from underthesea import ner

# Cấu hình API Google Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyA-Lymc-kEm_p00YMcQEbAlvgO79ZO-fgQ")
genai.configure(api_key=GOOGLE_API_KEY)

SIM_THRESHOLD = 0.55
output_image = "D:/LuanVanTotNghiep/NhanDienThongTinTranh/demo.jpg"
EXCLUDED_DOMAINS = [
    "google.com", "accounts.google.com", "policies.google.com", "support.google.com",
    "myactivity.google.com", "www.google.com.vn", "www.google.com"
]


def is_valid_article_url(img_url):
    if not img_url:
        return False
        # Loại bỏ ảnh base64
    if img_url.startswith("data:image"):
        return False
        # Loại bỏ ảnh có kích thước nhỏ (logo, icon)
    if "logo" in img_url.lower() or "icon" in img_url.lower():
        return False
    return True


def save_temp_image(image):
    """Lưu mảng ảnh NumPy vào thư mục chỉ định và trả về đường dẫn."""
    save_dir = "D:/LuanVanTotNghiep/NhanDienThongTinTranh/"
    os.makedirs(save_dir, exist_ok=True)  # Tạo thư mục nếu chưa tồn tại

    filename = f"temp_{uuid.uuid4().hex}.jpg"  # Tạo tên file ngẫu nhiên
    temp_file_path = os.path.join(save_dir, filename)

    cv2.imwrite(temp_file_path, image)  # Lưu ảnh vào file
    return temp_file_path


def preprocess_image(image, mode='gray'):
    """
    Hàm tiền xử lý ảnh để áp dụng các phương pháp lọc.
    - mode='gray' sẽ chuyển ảnh sang grayscale
    - mode='rgb' sẽ giữ ảnh nguyên trạng nếu là ảnh RGB
    """
    if mode == 'gray':
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Chuyển sang grayscale
    elif mode == 'rgb' and len(image.shape) == 2:  # Chuyển từ grayscale sang RGB nếu cần
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif mode == 'hsv':
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return image


def histogram_matching(image1, image2):
    """
    So sánh histogram của ảnh theo không gian màu HSV và grayscale.
    """
    similarity_hsv = 0

    # Chuyển ảnh sang HSV nếu chưa ở HSV
    if len(image1.shape) == 3:  # Kiểm tra xem có phải ảnh màu không
        image1_hsv = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
        image2_hsv = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)
    else:
        raise ValueError("Ảnh đầu vào phải có 3 kênh màu BGR!")

    # So sánh histogram trong không gian HSV
    for i in range(3):  # H, S, V
        hist1 = cv2.calcHist([image1_hsv], [i], None, [256], [0, 256])
        hist2 = cv2.calcHist([image2_hsv], [i], None, [256], [0, 256])

        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()

        similarity_hsv += cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    similarity_hsv /= 3  # Lấy trung bình trên 3 kênh H, S, V

    # So sánh histogram trong không gian ảnh xám
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    hist1_gray = cv2.calcHist([image1_gray], [0], None, [256], [0, 256])
    hist2_gray = cv2.calcHist([image2_gray], [0], None, [256], [0, 256])

    hist1_gray = cv2.normalize(hist1_gray, hist1_gray).flatten()
    hist2_gray = cv2.normalize(hist2_gray, hist2_gray).flatten()

    similarity_gray = cv2.compareHist(hist1_gray, hist2_gray, cv2.HISTCMP_CORREL)

    # Trọng số cho mỗi loại (có thể điều chỉnh)
    alpha = 0.7  # HSV quan trọng hơn màu xám
    beta = 0.3  # Ảnh xám có trọng số thấp hơn

    total_similarity = alpha * similarity_hsv + beta * similarity_gray

    return total_similarity


def ssim_compare(image1, image2):
    # Resize image2 theo kích thước của image1
    image2_resized = resize(image2, image1.shape, anti_aliasing=True)

    # Xác định giá trị win_size phù hợp
    min_dim = min(image1.shape[0], image1.shape[1])  # Chiều nhỏ nhất của ảnh
    win_size = min(7, min_dim)  # Giữ win_size <= chiều nhỏ nhất của ảnh

    # Chuyển về float64 nếu cần
    image1 = image1.astype(np.float64)
    image2_resized = image2_resized.astype(np.float64)

    # Xác định data_range
    data_range = image1.max() - image1.min()

    return ssim(image1, image2_resized, channel_axis=-1, win_size=win_size, data_range=data_range)


def sift_match(image1, image2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(image1, None)
    kp2, des2 = sift.detectAndCompute(image2, None)

    if des1 is None or des2 is None:
        return 0

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    return len(good_matches) / min(len(kp1), len(kp2))


def dino_similarity(image1, image2):
    # Gọi hàm DINOv2 để trích xuất đặc trưng ảnh (cần có mô hình DINOv2 cài sẵn)
    features1 = extract_dino_features(image1)
    features2 = extract_dino_features(image2)
    if features1 is None or features2 is None:
        print("❌ DINO không trích xuất được đặc trưng.")
    else:
        cosine_sim = np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))
        print(f"🧠 DINO Cosine Similarity: {cosine_sim}")
        return cosine_sim


def process_image_for_search(image_path):
    try:
        crop_painting(image_path, output_image)
        image = cv2.imread(output_image)  # Đọc ảnh màu
        image_rgb = preprocess_image(image, mode='rgb')  # Giữ nguyên ảnh RGB
        image_gray = preprocess_image(image, mode='gray')  # Chuyển ảnh sang grayscale
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Chuyển ảnh sang HSV

        print(f"Image loaded: {image.shape}")

        # Tiến hành lọc ảnh
        best_image = None
        images_to_process = "D:/LuanVanTotNghiep/NhanDienThongTinTranh/temp_images"
        image_files = [os.path.join(images_to_process, f) for f in os.listdir(images_to_process)]

        images = []
        for img_path in image_files:
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
            else:
                print(f"❌ Không thể đọc ảnh: {img_path}")

        # 1. Lọc Histogram kết hợp HSV + Grayscale
        for img in images:
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hist_score = histogram_matching(image_hsv, img_hsv)
            print(f"📊 Histogram similarity: {hist_score}")

            if hist_score > SIM_THRESHOLD:
                # 2. Lọc SSIM (trên ảnh RGB)
                ssim_score = ssim_compare(image_rgb, img)
                print(f"📈 SSIM Score: {ssim_score}")

                if ssim_score >= SIM_THRESHOLD:
                    best_image = img
                    break

        # Nếu không có ảnh nào tốt, thực hiện bước SIFT trên ảnh Grayscale
        sift_candidates = []

        if best_image is None:
            for img in images:
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Chuyển về grayscale nếu cần
                sift_score = sift_match(image_gray, img_gray)
                print(f"🔍 SIFT Score: {sift_score}")

                if sift_score > SIM_THRESHOLD:
                    sift_candidates.append((img, sift_score))

        # Nếu chỉ có đúng 1 ảnh đạt ngưỡng SIFT, chọn ngay
        if len(sift_candidates) == 1:
            best_image = sift_candidates[0][0]
        elif len(sift_candidates) > 1:
            images = [img for img, _ in sift_candidates]

        # Nếu không tìm được ảnh tốt, dùng DINOv2 trên ảnh RGB
        if best_image is None:
            max_dino_score = -1
            for img in images:
                dino_score = dino_similarity(image, img)
                if dino_score > max_dino_score:
                    max_dino_score = dino_score
                    best_image = img

        if best_image is None:
            print("Không tìm được ảnh phù hợp.")
            return None

        temp_image_path = save_temp_image(best_image)
        print(temp_image_path)

        # Tìm kiếm lại trên Google và lấy nội dung bài báo
        result = search_google_and_extract_info(temp_image_path)
        return result
    finally:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
            print(f"Đã xóa file: {temp_image_path}")
        else:
            print(f"File {temp_image_path} không tồn tại.")


def extract_info_from_article(page_url):
    try:
        response = requests.get(page_url, timeout=10)
        if response.status_code != 200:
            return {"description": "Không rõ", "source_page": page_url}

        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        description = " ".join([p.text.strip() for p in paragraphs]) if paragraphs else "Không rõ"

        return {"description": description, "source_page": page_url}
    except Exception as e:
        return {"description": f"Lỗi phân tích bài báo: {e}", "source_page": page_url}


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def extract_info_from_gemini(image_path, articles):
    try:
        # Ghép nối tất cả các mô tả từ bài viết
        description_text = " ".join([article["description"] for article in articles])
        image_data = encode_image(image_path)  # Hàm này cần phải định nghĩa để mã hóa ảnh

        # Tạo prompt cho Gemini
        prompt = (
            f"Phân tích bức tranh từ hình ảnh và thông tin bài viết dưới đây: {description_text}. "
            "Trả lời chi tiết và cụ thể về các thông tin sau: "
            "- Tên bức tranh. "
            "- Tên nghệ sĩ (nếu có). "
            "- Phong cách và các đặc điểm nghệ thuật nổi bật của tác phẩm. "
            "- Thể loại của bức tranh (ví dụ: tranh ngựa, phong cảnh, v.v.). "
            "- Năm sáng tác (nếu có thông tin). "
            "- Mô tả chi tiết về bức tranh, các đặc điểm nổi bật như màu sắc, hình dáng, và bố cục. "
            "- Các yếu tố đặc biệt hoặc thông tin bổ sung (ví dụ: ký hiệu, chữ ký, v.v.). "
            "Trả kết quả theo định dạng JSON nếu có thể. Nếu không, trả lời dưới dạng văn bản mô tả chi tiết."
        )

        # Khởi tạo mô hình Gemini
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        response = model.generate_content(
            contents=[{"mime_type": "image/jpeg", "data": image_data}, prompt]
        )

        result = response.text if hasattr(response, 'text') else response.content if hasattr(response,
                                                                                             'content') else ""

        # In kết quả trả về từ Gemini để debug
        print("Result from Gemini:", result)

        # Kiểm tra xem response có hợp lệ không
        if not result:
            return {"error": "Không có kết quả trả về từ Gemini."}

        # Định nghĩa các từ khóa chuẩn mà bạn muốn trích xuất
        predefined_keywords = {
            "title": ["title", "painting_title", "name_of_painting", "tên_bức_tranh"],  # Thêm từ khóa cho tên bức tranh
            "artist": ["artist", "name", "author", "tên_nghệ_sĩ"],
            "style": ["style", "artistic_style", "genre_style", "phong_cách"],
            "genre": ["genre", "type", "category", "thể_loại"],
            "year": ["year", "year_of_creation", "creation_year", "năm_sáng_tác"],
            "description": ["description", "details", "painting_description", "mô_tả"],
            "artistic_features": ["artistic_features", "special_features", "painting_features", "characteristics",
                                  "đặc_điểm_nghệ_thuật"],
            "additional_information": ["additional_information", "extra_info", "additional_details",
                                       "additional_info", "yếu_tố_đặc_biệt"]
        }

        # Cố gắng phân tích kết quả dưới dạng JSON
        try:
            # Kiểm tra nếu kết quả là một chuỗi JSON hợp lệ
            result_json = json.loads(result)

            # Khớp các từ khóa với các khóa trong JSON
            extracted_info = {}
            for key, possible_keys in predefined_keywords.items():
                for possible_key in possible_keys:
                    if possible_key in result_json:
                        extracted_info[key] = result_json[possible_key]
                        break
                # Nếu không tìm thấy bất kỳ khóa nào, gán giá trị mặc định
                if key not in extracted_info:
                    extracted_info[key] = "Không có thông tin"

            return extracted_info  # Trả về kết quả JSON nếu phân tích thành công
        except json.JSONDecodeError:
            # Nếu không phải JSON, sử dụng regex để trích xuất thông tin cần thiết
            extracted_info = {
                "title": "Không có thông tin",  # Thêm giá trị mặc định cho tên bức tranh
                "artist": "Không có thông tin",
                "style": "Không có thông tin",
                "genre": "Không có thông tin",
                "year": "Không có thông tin",
                "description": "Không có thông tin",
                "artistic_features": "Không có thông tin",
                "additional_info": "Không có thông tin"
            }

            # Sử dụng regex để trích xuất các thông tin khác
            extracted_info["title"] = re.search(r"\"title\"\s*:\s*\"([^\"]+)\"", result) or re.search(
                r"\"painting_title\"\s*:\s*\"([^\"]+)\"", result) or re.search(
                r"\"name_of_painting\"\s*:\s*\"([^\"]+)\"", result) or re.search(
                r"\"tên_bức_tranh\"\s*:\s*\"([^\"]+)\"", result)
            extracted_info["artist"] = re.search(r"\"artist\"\s*:\s*\"([^\"]+)\"", result) or re.search(
                r"\"name\"\s*:\s*\"([^\"]+)\"", result) or re.search(r"\"author\"\s*:\s*\"([^\"]+)\"",
                                                                     result) or re.search(
                r"\"tên_nghệ_sĩ\"\s*:\s*\"([^\"]+)\"", result)
            extracted_info["style"] = re.search(r"\"style\"\s*:\s*\"([^\"]+)\"", result) or re.search(
                r"\"artistic_style\"\s*:\s*\"([^\"]+)\"", result) or re.search(r"\"phong_cách\"\s*:\s*\"([^\"]+)\"",
                                                                               result)
            extracted_info["genre"] = re.search(r"\"genre\"\s*:\s*\"([^\"]+)\"", result) or re.search(
                r"\"type\"\s*:\s*\"([^\"]+)\"", result) or re.search(r"\"thể_loại\"\s*:\s*\"([^\"]+)\"", result)
            extracted_info["year"] = re.search(r"\"year_of_creation\"\s*:\s*\"([^\"]+)\"", result) or re.search(
                r"\"year\"\s*:\s*\"([^\"]+)\"", result) or re.search(r"\"creation_year\"\s*:\s*\"([^\"]+)\"",
                                                                     result) or re.search(
                r"\"năm_sáng_tác\"\s*:\s*\"([^\"]+)\"", result)
            extracted_info["description"] = re.search(r"\"description\"\s*:\s*\"([^\"]+)\"", result) or re.search(
                r"\"mô_tả\"\s*:\s*\"([^\"]+)\"", result) or re.search(r"\"mô_tả_chi_tiết\"\s*:\s*\"([^\"]+)\"", result)
            extracted_info["artistic_features"] = re.search(r"\"artistic_features\"\s*:\s*\[([^\]]+)\]",
                                                            result) or re.search(
                r"\"characteristics\"\s*:\s*\[([^\]]+)\]", result) or re.search(
                r"\"đặc_điểm_nghệ_thuật\"\s*:\s*\[([^\]]+)\]", result) or re.search(
                r"\"đặc_điểm_nghệ_thuật_nổi_bật\"\s*:\s*\[([^\]]+)\]", result) or re.search(
                r"\"artistic_features\"\s*:\s*\"([^\"]+)\"", result) or re.search(
                r"\"characteristics\"\s*:\s*\"([^\"]+)\"", result) or re.search(
                r"\"đặc_điểm_nghệ_thuật\"\s*:\s*\"([^\"]+)\"", result) or re.search(
                r"\"đặc_điểm_nghệ_thuật_nổi_bật\"\s*:\s*\"([^\"]+)\"", result)
            extracted_info["additional_info"] = re.search(r"\"additional_information\"\s*:\s*\[([^\]]+)\]",
                                                          result) or re.search(
                r"\"additional_info\"\s*:\s*\[([^\]]+)\]", result) or re.search(
                r"\"yếu_tố_đặc_biệt\"\s*:\s*\[([^\]]+)\]", result) or re.search(
                r"\"yếu_tố_đặc_biệt_hoặc_thông_tin_bổ_sung\"\s*:\s*\[([^\]]+)\]", result) or re.search(
                r"\"additional_information\"\s*:\s*\"([^\"]+)\"", result) or re.search(
                r"\"additional_info\"\s*:\s*\"([^\"]+)\"", result) or re.search(
                r"\"yếu_tố_đặc_biệt\"\s*:\s*\"([^\"]+)\"", result) or re.search(
                r"\"yếu_tố_đặc_biệt_hoặc_thông_tin_bổ_sung\"\s*:\s*\"([^\"]+)\"", result)

            # Chuyển đổi kết quả regex thành giá trị thực tế
            extracted_info = {
                k: v.group(1).strip() if v else "Không có thông tin" for k, v in extracted_info.items()
            }

            # Trả về kết quả dưới dạng văn bản mô tả chi tiết
            detailed_description = (
                f"Thông tin về bức tranh:\n"
                f"1. Tên bức tranh: {extracted_info['title']}\n"  # Thêm tên bức tranh
                f"2. Tên nghệ sĩ: {extracted_info['artist']}\n"
                f"3. Phong cách và đặc điểm nghệ thuật: {extracted_info['style']}\n"
                f"4. Thể loại: {extracted_info['genre']}\n"
                f"5. Năm sáng tác: {extracted_info['year']}\n"
                f"6. Mô tả: {extracted_info['description']}\n"
                f"7. Các đặc điểm nghệ thuật nổi bật: {extracted_info['artistic_features']}\n"
                f"8. Thông tin bổ sung: {extracted_info['additional_info']}\n"
            )

            # Kiểm tra kết quả cuối cùng
            if all(value == "Không có thông tin" for value in extracted_info.values()):
                # Nếu tất cả các trường không có thông tin, trả về văn bản mô tả chi tiết
                return detailed_description
            else:
                # Nếu có bất kỳ thông tin nào hợp lệ, trả về thông tin đã trích xuất
                return extracted_info

    except Exception as e:
        return {"error": f"Lỗi khi trích xuất thông tin từ Gemini: {str(e)}"}


def search_google_and_extract_info(image_path):
    print("🔍 Đang tìm kiếm bài báo từ Google Image...")

    chrome_options = uc.ChromeOptions()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-popup-blocking")
    chrome_options.add_argument("--disable-notifications")
    chrome_options.add_argument("--disable-infobars")
    chrome_options.add_argument("--disable-backgrounding-occluded-windows")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

    driver = uc.Chrome(options=chrome_options)
    driver.execute_script("""
        Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
        Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
        Object.defineProperty(navigator, 'platform', {get: () => 'Win32'});""")

    try:
        driver.get("https://www.google.com/?hl=vi")
        driver.minimize_window()
        time.sleep(2)

        # Click vào nút tìm kiếm bằng hình ảnh
        camera_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//div[@aria-label="Tìm kiếm bằng hình ảnh"]'))
        )
        camera_button.click()
        time.sleep(1)

        # Upload ảnh để tìm kiếm
        file_input = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, '//input[@type="file"]'))
        )
        abs_path = os.path.abspath(image_path)
        file_input.send_keys(abs_path)
        time.sleep(5)

        # Chuyển sang tab "Kết quả khớp chính xác"
        try:
            about_tab = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, '//div[contains(text(), "Kết quả khớp chính xác")]')))
            about_tab.click()
            time.sleep(1)

            article_links = driver.find_elements(By.XPATH, '//div[@id="search"]//a[contains(@href, "http")]')
            matched_articles = []

            for link in article_links:
                url = link.get_attribute("href")
                if not url or not is_valid_article_url(url):
                    continue

                article_info = extract_info_from_article(url)
                matched_articles.append(article_info)
                if len(matched_articles) >= 5:
                    break

            gemini_info = extract_info_from_gemini(image_path, matched_articles) if matched_articles else {}

            if matched_articles:
                result = {
                    "source": "Google Image",
                    "gemini_info": gemini_info
                }
                driver.quit()
                return result
        except Exception as e:
            print(f"❌ Lỗi: {e}")

        try:
            img_tab = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, '//div[contains(text(), "Hình ảnh trùng khớp")]')))
            img_tab.click()
            time.sleep(2)

            images = driver.find_elements(By.XPATH, '//div[contains(@class, "gdOPf")]//img')
            if not images:
                print("❌ Không tìm thấy ảnh nào.")
                driver.quit()
                return {"error": "Không tìm thấy ảnh phù hợp."}

            images_to_download = images[:10]
            os.makedirs("temp_images", exist_ok=True)
            action = ActionChains(driver)

            for idx, img in enumerate(images_to_download):
                try:
                    action.move_to_element(img).click().perform()
                    time.sleep(2)
                    large_images = driver.find_elements(By.XPATH, '//div[contains(@class, "p7sI2 PUxBg")]//img')
                    img_url = next((img.get_attribute("src") for img in large_images if img.get_attribute("src")), None)
                    if img_url:
                        with open(f"temp_images/image_{idx + 1}.jpg", "wb") as f:
                            f.write(requests.get(img_url).content)
                        print(f"✅ Đã tải ảnh {idx + 1}: {img_url}")
                except Exception as e:
                    print(f"⚠️ Lỗi khi tải ảnh {idx + 1}: {e}")

            driver.quit()

        except Exception as e:
            print(f"❌ Lỗi khi chuyển tab hình ảnh trùng khớp: {e}")
            driver.quit()
            return {"error": "Không thể chuyển tab hình ảnh trùng khớp."}

    except Exception as ex:
        driver.quit()
        return {"error": f"Lỗi khi tìm kiếm: {ex}"}

