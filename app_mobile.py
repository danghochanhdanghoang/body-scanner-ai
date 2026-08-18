import math
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image

# 1. Cấu hình giao diện
st.set_page_config(
    page_title="AI Body Scanner", page_icon="👗", layout="centered"
)
st.title("👗 AI Quét Dáng & Gợi Ý Phối Đồ")
st.write(
    "Tải ảnh toàn thân để AI đo ước tính, sau đó bạn có thể tự do điều chỉnh"
    " số đo chuẩn xác!"
)

# 2. Khởi tạo MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


# 3. Hàm tính khoảng cách pixel
def calculate_distance(p1, p2, width, height):
  x1, y1 = p1.x * width, p1.y * height
  x2, y2 = p2.x * width, p2.y * height
  return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# 4. Form nhập liệu
uploaded_file = st.file_uploader(
    "Chọn hoặc chụp ảnh toàn thân (JPG, PNG)", type=["jpg", "jpeg", "png"]
)
real_height_cm = st.number_input(
    "Nhập chiều cao thực tế của bạn (cm):",
    min_value=100.0,
    max_value=250.0,
    value=160.0,
    step=1.0,
)

if uploaded_file is not None:
  image = Image.open(uploaded_file)
  image_np = np.array(image)
  image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
  height, width, _ = image_rgb.shape

  results = pose.process(image_rgb)

  if results.pose_landmarks:
    annotated_image = image_rgb.copy()
    landmark_style = mp_drawing.DrawingSpec(
        color=(0, 255, 0), thickness=2, circle_radius=2
    )
    connection_style = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=landmark_style,
        connection_drawing_spec=connection_style,
    )

    st.image(
        annotated_image,
        channels="RGB",
        caption="Khung xương AI nhận diện",
        use_container_width=True,
    )

    # --- ƯỚC TÍNH SỐ ĐO AI ---
    landmarks = results.pose_landmarks.landmark
    nose = landmarks[mp_pose.PoseLandmark.NOSE]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

    # Tính chiều cao Pixel
    mid_ankle_y = (left_ankle.y + right_ankle.y) / 2
    pixel_height = abs(mid_ankle_y - nose.y) * height * 1.15
    cm_per_pixel = real_height_cm / pixel_height if pixel_height > 0 else 0

    # AI tính số đo thô (cộng hệ số bù độ dày thịt & quần áo)
    raw_shoulder = (
        calculate_distance(left_shoulder, right_shoulder, width, height)
        * cm_per_pixel
        * 1.25
    )
    raw_hip = (
        calculate_distance(left_hip, right_hip, width, height)
        * cm_per_pixel
        * 1.20
    )
    raw_waist = raw_hip * 0.80


    # Làm tròn số nguyên an toàn trong khoảng (1 - 200)
    def clean_val(val):
      if math.isnan(val) or val <= 0:
        return 38
      return int(max(1, min(200, round(val))))

    calc_shoulder = clean_val(raw_shoulder)
    calc_waist = clean_val(raw_waist)
    calc_hip = clean_val(raw_hip)

    st.success(
        "✅ Đã quét xong! Bạn có thể chỉnh lại số đo bên dưới nếu AI tính bị"
        " lệch."
    )

    # --- KHU VỰC TINH CHỈNH (TRỰC TIẾP KHÔNG DÙNG KEY ĐỂ TRÁNH XUNG ĐỘT STATE) ---
    st.subheader("📏 Tinh chỉnh số đo RỘNG VAI / EO / HÔNG (cm)")
    col1, col2, col3 = st.columns(3)
    with col1:
      final_shoulder = st.number_input(
          "Rộng Vai",
          value=calc_shoulder,
          min_value=1,
          max_value=200,
          step=1,
      )
    with col2:
      final_waist = st.number_input(
          "Rộng Eo",
          value=calc_waist,
          min_value=1,
          max_value=200,
          step=1,
      )
    with col3:
      final_hip = st.number_input(
          "Rộng Hông",
          value=calc_hip,
          min_value=1,
          max_value=200,
          step=1,
      )

    # --- KẾT QUẢ PHÂN TÍCH ---
    st.markdown("---")
    st.header("✨ Kết Quả Phân Tích & Gợi Ý")

    if final_waist > final_shoulder and final_waist > final_hip:
      shape = "Dáng Quả Táo (Apple)"
      advice = (
          "- **Nên mặc:** Áo cổ chữ V, đầm chữ A, quần cạp cao để tạo hiệu ứng"
          " eo thon.\n- **Tránh mặc:** Áo bó sát vòng 2, thắt lưng bản to."
      )
    elif final_hip > final_shoulder * 1.05:
      shape = "Dáng Quả Lê (Pear)"
      advice = (
          "- **Nên mặc:** Áo trễ vai, bèo nhún phần ngực, quần ống suông/tối"
          " màu để cân bằng phần hông.\n- **Tránh mặc:** Quần skinny sáng màu,"
          " váy xếp ly xòe quá rộng."
      )
    elif final_shoulder > final_hip * 1.05:
      shape = "Dáng Tam Giác Ngược"
      advice = (
          "- **Nên mặc:** Váy chữ A, váy xòe, quần ống rộng, áo cổ chữ V đơn"
          " giản.\n- **Tránh mặc:** Áo độn vai, áo trễ vai ngang, cổ thuyền."
      )
    elif abs(final_shoulder - final_hip) < (
        final_shoulder * 0.08
    ) and final_waist < (final_hip * 0.80):
      shape = "Dáng Đồng Hồ Cát (Hourglass)"
      advice = (
          "- **Nên mặc:** Đầm ôm body, áo croptop, quần/váy cạp cao tôn eo, thắt"
          " lưng điểm nhấn.\n- **Tránh mặc:** Quần áo oversize rộng thùng thình"
          " làm giấu đi đường cong."
      )
    else:
      shape = "Dáng Chữ Nhật (Rectangle)"
      advice = (
          "- **Nên mặc:** Áo có điểm nhấn ở eo (thắt nơ, đai), váy xòe bồng"
          " bềnh, họa tiết nổi bật.\n- **Tránh mặc:** Trang phục suông tuột từ"
          " trên xuống dưới."
      )

    st.subheader(f"📍 Vóc dáng của bạn: {shape}")
    st.write(advice)

  else:
    st.error(
        "❌ Không tìm thấy người trong ảnh. Vui lòng chụp rõ toàn thân và thử"
        " lại!"
    )
