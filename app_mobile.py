import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

# --- HÀM PHÂN LOẠI DÁNG NGƯỜI (Giữ nguyên) ---
def classify_body_shape(shoulder_cm, waist_cm, hip_cm):
    if shoulder_cm <= 0 or waist_cm <= 0 or hip_cm <= 0:
        return "Uncertain", "Không đủ dữ liệu."
    
    max_sh_hip = max(shoulder_cm, hip_cm)
    min_sh_hip = min(shoulder_cm, hip_cm)
    
    if (waist_cm / shoulder_cm >= 0.85) or (waist_cm / hip_cm >= 0.85):
        return "Apple (Dáng Quả Táo)", "Phần thân trên và eo đầy đặn. Gợi ý: Trang phục cổ chữ V, đầm dáng xòe A-line nhẹ, tránh thắt lưng quá chặt."
    elif (shoulder_cm / hip_cm) >= 1.05:
        return "Inverted Triangle (Tam Giác Ngược)", "Vai rộng, hông hẹp. Gợi ý: Quần ống rộng, chân váy xòe, áo đơn giản tối màu ở phần trên."
    elif (hip_cm / shoulder_cm) >= 1.05:
        return "Pear (Dáng Quả Lê)", "Hông nở, vai nhỏ. Gợi ý: Áo có bèo nhún/đệm vai, quần hoặc chân váy suông thẳng tối màu."
    else:
        if (waist_cm / min_sh_hip) <= 0.75:
            return "Hourglass (Dáng Đồng Hồ Cát)", "Tỷ lệ chuẩn với đường thắt eo rõ. Gợi ý: Đầm ôm sát, áo chiết eo, thắt lưng tôn dáng."
        else:
            return "Rectangle (Dáng Chữ Nhật)", "Thân hình thẳng, đường thắt eo ít rõ. Gợi ý: Tạo điểm nhấn eo bằng thắt lưng, chân váy xếp ly, đầm xòe."

# --- HÀM XỬ LÝ ẢNH CHÍNH ---
def process_image(image_np, user_height_cm):
    mp_pose = mp.solutions.pose
    h, w, _ = image_np.shape

    with mp_pose.Pose(
        static_image_mode=True, model_complexity=1, enable_segmentation=True, min_detection_confidence=0.6
    ) as pose:
        results = pose.process(image_np)
        
        if not results.pose_landmarks:
            return None, "Không phát hiện được người trong ảnh. Vui lòng chụp rõ toàn thân!"

        mask = (results.segmentation_mask > 0.5).astype(np.uint8) * 255
        y_indices = np.where(mask > 0)[0]
        if len(y_indices) == 0:
            return None, "Không trích xuất được phom dáng."
            
        total_height_px = np.max(y_indices) - np.min(y_indices)
        px_to_cm = user_height_cm / total_height_px if total_height_px > 0 else 0

        landmarks = results.pose_landmarks.landmark
        l_shoulder = np.array([int(landmarks[11].x * w), int(landmarks[11].y * h)])
        r_shoulder = np.array([int(landmarks[12].x * w), int(landmarks[12].y * h)])
        l_hip = np.array([int(landmarks[23].x * w), int(landmarks[23].y * h)])
        r_hip = np.array([int(landmarks[24].x * w), int(landmarks[24].y * h)])

        # 🚀 THUẬT TOÁN MỚI: Quét từ trục xương sống ra 2 bên, chạm viền (gap) là DỪNG.
        def get_body_width_from_center(y_coord, center_x):
            if y_coord < 0 or y_coord >= h: return 0, 0, 0
            row = mask[y_coord, :]
            cx = int(center_x)
            
            # Nếu tâm điểm bị rỗng (ngoài mask), bỏ qua
            if cx < 0 or cx >= w or row[cx] == 0: 
                return 0, 0, 0
            
            # Dò sang trái đến khi chạm rìa người (pixel đen)
            x1 = cx
            while x1 > 0 and row[x1] > 0:
                x1 -= 1
                
            # Dò sang phải đến khi chạm rìa người (pixel đen)
            x2 = cx
            while x2 < w - 1 and row[x2] > 0:
                x2 += 1
                
            return (x2 - x1), x1, x2

        # Lấy tọa độ trung tâm (Xương sống)
        y_shoulder_avg = int((l_shoulder[1] + r_shoulder[1]) / 2)
        y_hip_avg = int((l_hip[1] + r_hip[1]) / 2)
        x_sh_center = (l_shoulder[0] + r_shoulder[0]) / 2
        x_hip_center = (l_hip[0] + r_hip[0]) / 2

        # 1. Đo Vai (Rộng nhất quanh vùng vai)
        max_sh_px, sh_y, sh_x1, sh_x2 = 0, -1, 0, 0
        for y in range(max(0, y_shoulder_avg - int(h*0.05)), min(h, y_shoulder_avg + int(h*0.05))):
            width, x1, x2 = get_body_width_from_center(y, x_sh_center)
            if width > max_sh_px: max_sh_px, sh_y, sh_x1, sh_x2 = width, y, x1, x2
        shoulder_cm = round(max_sh_px * px_to_cm, 1)

        # 2. Đo Eo (Nhỏ nhất giữa ngực và bụng)
        min_waist_px, waist_y, waist_x1, waist_x2 = float('inf'), -1, 0, 0
        for y in range(y_shoulder_avg + int((y_hip_avg - y_shoulder_avg) * 0.30), y_shoulder_avg + int((y_hip_avg - y_shoulder_avg) * 0.80)):
            # Tịnh tiến trục tâm theo xương sống (chống lệch nếu người đứng nghiêng)
            progress = (y - y_shoulder_avg) / (y_hip_avg - y_shoulder_avg + 1e-6)
            cx = x_sh_center + (x_hip_center - x_sh_center) * progress
            
            width, x1, x2 = get_body_width_from_center(y, cx)
            if 0 < width < min_waist_px: min_waist_px, waist_y, waist_x1, waist_x2 = width, y, x1, x2
        
        if min_waist_px == float('inf'): min_waist_px = 0
        waist_cm = round(min_waist_px * px_to_cm, 1)

        # 3. Đo Hông (Rộng nhất quanh vùng hông/mông)
        max_hip_px, hip_y, hip_x1, hip_x2 = 0, -1, 0, 0
        for y in range(y_shoulder_avg + int((y_hip_avg - y_shoulder_avg) * 0.80), min(h - 1, y_shoulder_avg + int((y_hip_avg - y_shoulder_avg) * 1.40))):
            width, x1, x2 = get_body_width_from_center(y, x_hip_center)
            if width > max_hip_px: max_hip_px, hip_y, hip_x1, hip_x2 = width, y, x1, x2
        hip_cm = round(max_hip_px * px_to_cm, 1)

        # Phân loại & Vẽ trực quan
        shape_name, advice = classify_body_shape(shoulder_cm, waist_cm, hip_cm)
        annotated_img = image_np.copy()
        
        # Vẽ lại 3 đường chuẩn xác dựa trên rìa cơ thể
        if sh_y != -1: cv2.line(annotated_img, (sh_x1, sh_y), (sh_x2, sh_y), (0, 255, 0), 4) # Xanh lá - Vai
        if waist_y != -1: cv2.line(annotated_img, (waist_x1, waist_y), (waist_x2, waist_y), (255, 0, 0), 4) # Xanh dương - Eo
        if hip_y != -1: cv2.line(annotated_img, (hip_x1, hip_y), (hip_x2, hip_y), (0, 0, 255), 4) # Đỏ - Hông

        result_data = {
            "shoulder": shoulder_cm, "waist": waist_cm, "hip": hip_cm,
            "shape": shape_name, "advice": advice
        }
        return annotated_img, result_data

# --- GIAO DIỆN WEB VỚI STREAMLIT (Giữ nguyên) ---
st.set_page_config(page_title="AI Stylist - Đo tỷ lệ cơ thể", layout="centered")

st.title("👗 AI Stylist - Phân tích dáng người")
st.write("Upload ảnh toàn thân hoặc chụp trực tiếp để AI tư vấn cách phối đồ cho bạn!")

user_height = st.number_input("Nhập chiều cao của bạn (cm):", min_value=100.0, max_value=250.0, value=165.0, step=1.0)

# Nút tải ảnh hỗ trợ cả chụp từ camera điện thoại
uploaded_file = st.file_uploader("Chọn ảnh hoặc chụp ảnh mới", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Đọc ảnh từ file upload
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    
    st.image(image, caption="Ảnh gốc", use_column_width=True)
    
    if st.button("Phân tích ngay", type="primary"):
        with st.spinner("AI đang quét tỷ lệ cơ thể..."):
            annotated_img, result = process_image(image_np, user_height)
            
            if isinstance(result, str):
                st.error(result) # Báo lỗi nếu không thấy người
            else:
                st.success("Phân tích thành công!")
                
                # Hiển thị kết quả
                st.image(annotated_img, caption="Ảnh AI đã quét", use_column_width=True)
                
                st.subheader("📊 Kết quả đo lường")
                col1, col2, col3 = st.columns(3)
                col1.metric("Vai", f"{result['shoulder']} cm")
                col2.metric("Eo", f"{result['waist']} cm")
                col3.metric("Hông", f"{result['hip']} cm")
                
                st.subheader("✨ Dáng người của bạn")
                st.info(f"**{result['shape']}**")
                
                st.subheader("💡 Gợi ý phối đồ")
                st.write(result['advice'])
