import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image

# 1. Cấu hình giao diện đơn giản
st.set_page_config(page_title="AI Body Pixel Scanner", page_icon="📐", layout="centered")
st.title("📐 AI Quét Tỷ Lệ Cơ Thể (Pixel Perfect)")
st.markdown("---")

# 2. Khởi tạo MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

mp_segmentation = mp.solutions.selfie_segmentation
segmentation = mp_segmentation.SelfieSegmentation(model_selection=1)

# Hàm phóng tia từ trục giữa ra 2 bên để tìm viền cơ thể
def find_edges_from_center(mask, center_x, y):
    h, w = mask.shape
    if y < 0 or y >= h:
        return None, None
    
    cx = int(center_x)
    if cx < 0 or cx >= w or not mask[y, cx]:
        return None, None  # Tâm nằm ngoài viền người

    # Dò sang trái
    left_x = cx
    while left_x > 0 and mask[y, left_x]:
        left_x -= 1
        
    # Dò sang phải
    right_x = cx
    while right_x < w - 1 and mask[y, right_x]:
        right_x += 1
        
    return left_x, right_x

# 3. Form nhập liệu (Chụp trực tiếp KHÔNG DÙNG BROWSER / Tải ảnh lên)
st.subheader("📸 Bước 1: Chọn ảnh hoặc chụp trực tiếp")
input_method = st.radio(
    "Chọn phương thức nhập ảnh:",
    ["Chụp ảnh trực tiếp (Camera)", "Tải ảnh từ thiết bị"]
)

uploaded_file = None
if input_method == "Chụp ảnh trực tiếp (Camera)":
    # Mở camera trực tiếp trên màn hình, không qua popup file browser
    uploaded_file = st.camera_input("Đứng thẳng, dang nhẹ tay và chụp!")
else:
    uploaded_file = st.file_uploader("Chọn ảnh có sẵn (JPG, PNG)", type=["jpg", "jpeg", "png"])

# 4. Khu vực xử lý chính
if uploaded_file is not None:
    # Đọc ảnh và chuyển sang Numpy array
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    h, w, _ = image_np.shape

    # Chạy AI
    with st.spinner("AI đang quét viền cơ thể..."):
        pose_results = pose.process(image_np)
        seg_results = segmentation.process(image_np)

    if pose_results.pose_landmarks and seg_results.segmentation_mask is not None:
        # Tạo mặt nạ nhị phân (người = True, nền = False)
        mask = seg_results.segmentation_mask > 0.5
        
        # Lấy tọa độ xương để định hướng vùng tìm kiếm
        lm = pose_results.pose_landmarks.landmark
        l_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        l_hip = lm[mp_pose.PoseLandmark.LEFT_HIP]
        r_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP]

        # Tọa độ Y và X trung bình của vai và hông
        y_sh = int((l_sh.y + r_sh.y) / 2 * h)
        y_hip = int((l_hip.y + r_hip.y) / 2 * h)
        x_sh_center = (l_sh.x + r_sh.x) / 2 * w
        x_hip_center = (l_hip.x + r_hip.x) / 2 * w

        # --- THUẬT TOÁN QUÉT VAI, EO, HÔNG (PIXEL) ---
        
        # 1. TÌM VAI (MAX width quanh khu vực vai)
        best_sh_w = -1
        pts_sh = None
        for y in range(max(0, y_sh - int(h*0.05)), min(h, y_sh + int(h*0.05))):
            lx, rx = find_edges_from_center(mask, x_sh_center, y)
            if lx is not None and rx is not None:
                width = rx - lx
                if width > best_sh_w:
                    best_sh_w = width
                    pts_sh = ((lx, y), (rx, y))

        # 2. TÌM EO (MIN width giữa ngực và bụng)
        best_ws_w = float('inf')
        pts_ws = None
        for y in range(y_sh + int(h*0.12), y_hip - int(h*0.05)):
            progress = (y - y_sh) / (y_hip - y_sh + 1e-6)
            current_center_x = x_sh_center + (x_hip_center - x_sh_center) * progress
            
            lx, rx = find_edges_from_center(mask, current_center_x, y)
            if lx is not None and rx is not None:
                width = rx - lx
                if width < best_ws_w and width > 0:
                    best_ws_w = width
                    pts_ws = ((lx, y), (rx, y))

        # 3. TÌM HÔNG (MAX width quanh khu vực mông)
        best_hp_w = -1
        pts_hp = None
        for y in range(y_hip - int(h*0.02), min(h, y_hip + int(h*0.15))):
            lx, rx = find_edges_from_center(mask, x_hip_center, y)
            if lx is not None and rx is not None:
                width = rx - lx
                if width > best_hp_w:
                    best_hp_w = width
                    pts_hp = ((lx, y), (rx, y))

        # --- VẼ LÊN ẢNH ĐỂ THỂ HIỆN TRỰC QUAN ---
        annotated_image = image_np.copy()
        
        def draw_measurement(img, pts, color, label):
            if pts:
                p1, p2 = pts
                # Vẽ đường nối
                cv2.line(img, p1, p2, color, max(2, int(w*0.005)))
                # Vẽ 2 điểm ngoài cùng (chấm tròn to rõ)
                cv2.circle(img, p1, max(4, int(w*0.015)), color, -1)
                cv2.circle(img, p2, max(4, int(w*0.015)), color, -1)
                # Ghi text
                text_pos = (int((p1[0]+p2[0])/2) - 40, p1[1] - 10)
                cv2.putText(img, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, max(0.5, w*0.0015), color, max(1, int(w*0.003)))

        draw_measurement(annotated_image, pts_sh, (255, 50, 50), "VAI")   
        draw_measurement(annotated_image, pts_ws, (50, 255, 50), "EO")    
        draw_measurement(annotated_image, pts_hp, (50, 50, 255), "HONG")  

        st.markdown("---")
        st.subheader("🎯 Bước 2: Kết quả đo đạc")
        st.image(annotated_image, channels="RGB", use_container_width=True)

        st.markdown("### 📊 Thông số chi tiết (Pixel)")
        c1, c2, c3 = st.columns(3)
        c1.metric("🔴 Rộng Vai", f"{best_sh_w} px" if pts_sh else "N/A")
        c2.metric("🟢 Rộng Eo", f"{best_ws_w} px" if pts_ws else "N/A")
        c3.metric("🔵 Rộng Hông", f"{best_hp_w} px" if pts_hp else "N/A")

    else:
        st.error("❌ Không tìm thấy người trong ảnh. Vui lòng chụp rõ toàn thân.")
