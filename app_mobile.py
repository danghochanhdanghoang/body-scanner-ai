import math
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st

# Cấu hình giao diện Mobile
st.set_page_config(page_title="AI Body Scanner", page_icon="👗", layout="centered")

st.title("👗 AI Quét Dáng Người & Gợi Ý Phối Đồ")
st.write("Tải ảnh toàn thân để AI đo kích thước và phân tích vóc dáng!")

# Khởi tạo MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Form nhập dữ liệu trên điện thoại
uploaded_file = st.file_uploader(
    "Chọn hoặc chụp ảnh toàn thân", type=["jpg", "jpeg", "png"]
)
user_height = st.number_input(
    "Nhập chiều cao thực tế (cm)", min_value=100.0, max_value=220.0, value=162.0
)

if uploaded_file is not None:
    # Đọc file ảnh từ bộ nhớ
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = pose.process(image_rgb)

    if results.pose_landmarks:
        h, w, _ = image.shape
        landmarks = results.pose_landmarks.landmark

        # Lấy tọa độ mốc
        l_sh = (int(landmarks[11].x * w), int(landmarks[11].y * h))
        r_sh = (int(landmarks[12].x * w), int(landmarks[12].y * h))
        l_hip = (int(landmarks[23].x * w), int(landmarks[23].y * h))
        r_hip = (int(landmarks[24].x * w), int(landmarks[24].y * h))

        # Tính toán chiều cao pixel & quy đổi cm
        nose = (landmarks[0].x * w, landmarks[0].y * h)
        mid_ankle = (
            ((landmarks[27].x + landmarks[28].x) / 2) * w,
            ((landmarks[27].y + landmarks[28].y) / 2) * h,
        )
        body_height_px = math.dist(nose, mid_ankle) * 1.12
        px_per_cm = body_height_px / user_height

        # Kích thước đã áp dụng hệ số bù mép cơ thể
        shoulder_cm = (math.dist(l_sh, r_sh) / px_per_cm) * 1.23
        hip_cm = (math.dist(l_hip, r_hip) / px_per_cm) * 1.58
        ratio = shoulder_cm / hip_cm if hip_cm > 0 else 0

        # === TÍCH HỢP HƯỚNG 2: VẼ VẠCH ĐO TRỰC TIẾP LÊN Ô ẢNH ===
        annotated_image = image_rgb.copy()
        # Vẽ đường nối Vai (Màu xanh dương)
        cv2.line(annotated_image, l_sh, r_sh, (0, 120, 255), 4)
        cv2.circle(annotated_image, l_sh, 6, (0, 255, 0), -1)
        cv2.circle(annotated_image, r_sh, 6, (0, 255, 0), -1)

        # Vẽ đường nối Hông (Màu đỏ)
        cv2.line(annotated_image, l_hip, r_hip, (255, 0, 0), 4)
        cv2.circle(annotated_image, l_hip, 6, (0, 255, 0), -1)
        cv2.circle(annotated_image, r_hip, 6, (0, 255, 0), -1)

        # === HIỂN THỊ KẾT QUẢ TRÊN GIAO DIỆN MOBILE ===
        st.image(
            annotated_image,
            caption="Ảnh đã qua phân tích AI (Đường xanh: Vai | Đường đỏ: Hông)",
            use_container_width=True,
        )

        st.subheader("📊 Kết quả phân tích:")
        col1, col2, col3 = st.columns(3)
        col1.metric("Độ rộng Vai", f"{shoulder_cm:.1f} cm")
        col2.metric("Độ rộng Hông", f"{hip_cm:.1f} cm")
        col3.metric("Tỷ lệ Vai/Hông", f"{ratio:.2f}")

        # Phân loại dáng người
        if ratio > 1.20:
            st.warning("👉 **Dáng cơ thể: Tam giác ngược**")
            st.info(
                "💡 **Gợi ý trang phục:** Nên chọn áo cổ V, xòe phần thân dưới, chọn quần ống rộng/chân váy chữ A để tạo sự cân đối."
            )
        elif ratio < 0.90:
            st.warning("👉 **Dáng cơ thể: Quả lê**")
            st.info(
                "💡 **Gợi ý trang phục:** Nên chọn áo có điểm nhấn ở vai/ngực, mặc tối màu phần dưới để thu hút ánh nhìn lên trên."
            )
        else:
            st.success("👉 **Dáng cơ thể: Cân đối / Chữ nhật**")
            st.info(
                "💡 **Gợi ý trang phục:** Dễ phối đồ, nên thắt lưng hoặc chọn đồ chiết eo để tạo đường cong."
            )
    else:
        st.error("Không tìm thấy người trong hình, vui lòng thử lại ảnh khác!")
