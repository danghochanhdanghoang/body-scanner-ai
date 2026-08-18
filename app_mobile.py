import math
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image

# 1. Cấu hình giao diện
st.set_page_config(
    page_title="AI Body Scanner Pro", page_icon="👗", layout="centered"
)
st.title("👗 AI Quét Dáng Pro (Chống Quét Lỏ)")
st.markdown("---")
st.write(
    "AI quét bị lỏ? Đừng lo! Số đo AI chỉ là tham khảo ban đầu. "
    "**Sức mạnh thực sự nằm ở việc bạn tinh chỉnh số đo chính xác bên dưới** "
    "để nhận gợi ý chuẩn 100%!"
)

# 2. Khởi tạo MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


# 3. Hàm tính khoảng cách pixel (Làm tròn số thực an toàn)
def calculate_distance(p1, p2, width, height):
  x1, y1 = p1.x * width, p1.y * height
  x2, y2 = p2.x * width, p2.y * height
  dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
  if math.isnan(dist) or dist < 0:
    return 0
  return dist


# 4. Form nhập liệu
uploaded_file = st.file_uploader(
    "Bước 1: Chọn hoặc chụp ảnh toàn thân (JPG, PNG)", type=["jpg", "jpeg", "png"]
)
real_height_cm = st.number_input(
    "Nhập chiều cao thực tế của bạn (cm):",
    min_value=100.0,
    max_value=250.0,
    value=160.0,
    step=1.0,
)

if uploaded_file is not None:
  # Load and process image
  image = Image.open(uploaded_file)
  image_np = np.array(image)
  image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
  height, width, _ = image_rgb.shape

  # Chạy MediaPipe Pose
  results = pose.process(image_rgb)

  if results.pose_landmarks:
    # Vẽ landmarks lên ảnh để tham khảo
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
        caption="Khung xương AI nhận diện (Tham khảo vị trí khớp xương)",
        use_container_width=True,
    )

    # --- ƯỚC TÍNH SỐ ĐO AI (LÀM TRÒN AN TOÀN TRONG KHOẢNG 1-200) ---
    def clean_val(val):
      if math.isnan(val) or val <= 0:
        return 38
      return int(max(1, min(200, round(val))))

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

    calc_shoulder = clean_val(raw_shoulder)
    calc_waist = clean_val(raw_waist)
    calc_hip = clean_val(raw_hip)

    st.success(
        "✅ AI đã quét và ước tính xong! Ghi nhận vị trí khớp xương."
    )

    # --- KHU VỰC TINH CHỈNH SỐ ĐO CHÍNH (KEY KHÔNG DÙNG ĐỂ TRÁNH XUNG ĐỘT STATE) ---
    st.subheader("📏 Bước 2: Tinh chỉnh số đo RỘNG VAI / EO / HÔNG (cm)")
    st.write(
        "AI quét không chính xác độ rộng cơ thể? Hãy **nhập con số chuẩn xác nhất** vào ô này:"
    )
    col1, col2, col3 = st.columns(3)
    with col1:
      final_shoulder = st.number_input(
          "Rộng Vai (cm)",
          value=calc_shoulder,  # Số AI tính chỉ là giá trị khởi tạo
          min_value=1,
          max_value=200,
          step=1,
          key="ui_shoulder",  # Đặt key để Streamlit quản lý UI độc lập
      )
    with col2:
      final_waist = st.number_input(
          "Rộng Eo (cm)",
          value=calc_waist,
          min_value=1,
          max_value=200,
          step=1,
          key="ui_waist",
      )
    with col3:
      final_hip = st.number_input(
          "Rộng Hông (cm)",
          value=calc_hip,
          min_value=1,
          max_value=200,
          step=1,
          key="ui_hip",
      )

    # Hiển thị số đo hiện tại đang phân tích
    st.markdown("---")
    st.markdown(
        f"📊 **Số đo hiện tại:** Vai: **{final_shoulder}cm**, Eo: **{final_waist}cm**, Hông: **{final_hip}cm**"
    )

    # --- KẾT QUẢ PHÂN TÍCH & GỢI Ý ---
    st.header("✨ Bước 3: Phân Tích Dáng Người & Gợi Ý Phối Đồ")
    shape = "Chưa xác định"
    advice = ""

    if final_waist > final_shoulder and final_waist > final_hip:
      shape = "Dáng Quả Táo (Apple)"
      advice = (
          "- **Đặc điểm:** Vòng 2 là vòng lớn nhất.\n"
          "- **Nên mặc:** Áo cổ chữ V sâu, đầm chữ A, quần cạp cao, thắt lưng nhẹ dưới chân ngực.\n"
          "- **Tránh mặc:** Áo bó sát vòng 2, trang phục oversize rộng thùng thình."
      )
    elif final_hip > final_shoulder * 1.05:
      shape = "Dáng Quả Lê (Pear)"
      advice = (
          "- **Đặc điểm:** Hông rõ rệt là vòng lớn nhất, vai nhỏ hơn.\n"
          "- **Nên mặc:** Áo trễ vai, bèo nhún phần ngực, quần ống suông/tối màu, chân váy chữ A.\n"
          "- **Tránh mặc:** Quần skinny sáng màu, váy xếp ly xòe quá rộng làm lộ hông to."
      )
    elif final_shoulder > final_hip * 1.05:
      shape = "Dáng Tam Giác Ngược"
      advice = (
          "- **Đặc điểm:** Vai rõ rệt lớn nhất, hông nhỏ.\n"
          "- **Nên mặc:** Chân váy chữ A, váy xòe bồng, quần ống rộng, áo cổ chữ V đơn giản.\n"
          "- **Tránh mặc:** Áo độn vai, áo trễ vai ngang, cổ thuyền bản to."
      )
    elif abs(final_shoulder - final_hip) < (
        final_shoulder * 0.08
    ) and final_waist < (final_hip * 0.80):
      shape = "Dáng Đồng Hồ Cát (Hourglass)"
      advice = (
          "- **Đặc điểm:** Vai và Hông tương đương, Vòng Eo nhỏ rõ rệt.\n"
          "- **Nên mặc:** Đầm ôm body, áo croptop, quần/váy cạp cao tôn eo, thắt lưng làm điểm nhấn.\n"
          "- **Tránh mặc:** Quần áo oversize rộng thùng thình làm giấu đi đường cong."
      )
    else:
      shape = "Dáng Chữ Nhật (Rectangle)"
      advice = (
          "- **Đặc điểm:** Vai, Eo, Hông tương đương nhau.\n"
          "- **Nên mặc:** Áo có điểm nhấn ở eo (thắt nơ, đai), váy xòe, họa tiết nổi bật phần trên/dưới.\n"
          "- **Tránh mặc:** Trang phục suông tuột từ trên xuống dưới."
      )

    st.subheader(f"📍 Vóc dáng của bạn: **{shape}**")
    st.markdown(advice)

  else:
    st.error(
        "❌ Không tìm thấy người trong ảnh. Vui lòng chụp rõ toàn thân và thử lại!"
    )
