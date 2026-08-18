import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import math
from PIL import Image

# 1. Cấu hình giao diện
st.set_page_config(page_title="AI Body Scanner", page_icon="👗", layout="centered")
st.title("👗 AI Quét Dáng & Gợi Ý Phối Đồ")
st.write("Tải ảnh toàn thân, AI sẽ đo thử và bạn có thể tinh chỉnh lại để có kết quả chuẩn nhất!")

# 2. Khởi tạo MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 3. Hàm tính khoảng cách giữa 2 điểm (Pixel)
def calculate_distance(p1, p2, width, height):
    x1, y1 = p1.x * width, p1.y * height
    x2, y2 = p2.x * width, p2.y * height
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# 4. Form nhập liệu
uploaded_file = st.file_uploader("Chọn hoặc chụp ảnh toàn thân (JPG, PNG)", type=["jpg", "jpeg", "png"])
real_height_cm = st.number_input("Nhập chiều cao thực tế của bạn (cm):", min_value=100.0, max_value=250.0, value=160.0, step=1.0)

if uploaded_file is not None:
    # Đọc ảnh
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    
    # Chuyển đổi RGB cho MediaPipe
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    height, width, _ = image_rgb.shape
    
    # Phân tích AI
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        # Vẽ khung xương AI lên ảnh (Tuỳ chỉnh nét vẽ)
        annotated_image = image_rgb.copy()
        landmark_style = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
        connection_style = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        mp_drawing.draw_landmarks(
            annotated_image, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=landmark_style,
            connection_drawing_spec=connection_style
        )
        
        st.image(annotated_image, channels="RGB", caption="Khung xương AI nhận diện", use_column_width=True)
        
        # --- BẮT ĐẦU TÍNH TOÁN KÍCH THƯỚC (ƯỚC TÍNH AI) ---
        landmarks = results.pose_landmarks.landmark
        
        # Lấy tọa độ các điểm chuẩn
        nose = landmarks[mp_pose.PoseLandmark.NOSE]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        
        # Tính chiều cao Pixel (cộng thêm 10% bù cho phần đỉnh đầu và bàn chân)
        mid_ankle_y = (left_ankle.y + right_ankle.y) / 2
        pixel_height = (mid_ankle_y - nose.y) * height * 1.1 
        
        # Tỷ lệ chuyển đổi Pixel -> CM
        cm_per_pixel = real_height_cm / pixel_height if pixel_height > 0 else 0
        
        # AI Đo khoảng cách (cm)
        ai_shoulder_cm = calculate_distance(left_shoulder, right_shoulder, width, height) * cm_per_pixel
        ai_hip_cm = calculate_distance(left_hip, right_hip, width, height) * cm_per_pixel
        ai_waist_cm = ai_hip_cm * 0.85 # Eo thường khó bắt điểm, AI tạm tính = 85% Hông
        
        st.success("✅ Đã quét xong! Vui lòng tinh chỉnh lại số đo bên dưới nếu AI nhận diện sai do quần áo.")
        
        # --- KHU VỰC NGƯỜI DÙNG TINH CHỈNH SỐ ĐO ---
        st.subheader("📏 Tinh chỉnh số đo (cm)")
        col1, col2, col3 = st.columns(3)
        with col1:
            final_shoulder = st.number_input("Rộng Vai", value=float(ai_shoulder_cm), min_value=10.0, max_value=100.0, step=1.0)
        with col2:
            final_waist = st.number_input("Rộng Eo", value=float(ai_waist_cm), min_value=10.0, max_value=100.0, step=1.0)
        with col3:
            final_hip = st.number_input("Rộng Hông", value=float(ai_hip_cm), min_value=10.0, max_value=100.0, step=1.0)

        # --- PHÂN TÍCH VÓC DÁNG & GỢI Ý ---
        st.markdown("---")
        st.header("✨ Kết Quả Phân Tích & Gợi Ý")
        
        # Thuật toán phân loại dáng người cơ bản
        shape = "Chưa xác định"
        advice = ""
        
        if final_waist > final_shoulder and final_waist > final_hip:
            shape = "Dáng Quả Táo (Apple)"
            advice = "- **Nên mặc:** Áo cổ chữ V, đầm chữ A, quần cạp cao để tạo hiệu ứng eo thon.\n- **Tránh mặc:** Áo bó sát vòng 2, thắt lưng bản to."
        elif final_hip > final_shoulder * 1.05:
            shape = "Dáng Quả Lê (Pear)"
            advice = "- **Nên mặc:** Áo trễ vai, bèo nhún phần ngực, quần ống suông/tối màu để cân bằng phần hông.\n- **Tránh mặc:** Quần skinny sáng màu, váy xếp ly xòe quá rộng."
        elif final_shoulder > final_hip * 1.05:
            shape = "Dáng Tam Giác Ngược"
            advice = "- **Nên mặc:** Váy chữ A, váy xòe, quần ống rộng, áo cổ chữ V đơn giản.\n- **Tránh mặc:** Áo độn vai, áo trễ vai ngang, cổ thuyền."
        elif abs(final_shoulder - final_hip) < (final_shoulder * 0.05) and final_waist < (final_hip * 0.75):
            shape = "Dáng Đồng Hồ Cát (Hourglass)"
            advice = "- **Nên mặc:** Đầm ôm body, áo croptop, quần/váy cạp cao tôn eo, thắt lưng điểm nhấn.\n- **Tránh mặc:** Quần áo oversize rộng thùng thình làm giấu đi đường cong."
        else:
            shape = "Dáng Chữ Nhật (Rectangle)"
            advice = "- **Nên mặc:** Áo có điểm nhấn ở eo (thắt nơ, đai), váy xòe bồng bềnh, họa tiết nổi bật.\n- **Tránh mặc:** Trang phục suông tuột từ trên xuống dưới."

        st.subheader(f"📍 Vóc dáng của bạn: {shape}")
        st.write(advice)

    else:
        st.error("❌ Không tìm thấy người trong ảnh. Vui lòng chụp rõ toàn thân và thử lại!")
