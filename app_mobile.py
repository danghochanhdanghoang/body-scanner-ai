import math
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# 1. Cấu hình giao diện
st.set_page_config(page_title="AI Body Scanner Pro", page_icon="👗", layout="centered")
st.title("👗 AI Quét Dáng & Kéo Thả Trực Tiếp")
st.markdown("---")
st.write(
    "💡 **Cách dùng:** Tải ảnh lên, AI sẽ hiển thị ảnh của bạn cùng 3 đường kẻ "
    "(🔴 Đỏ: Vai, 🟢 Xanh lá: Eo, 🔵 Xanh dương: Hông). Hãy kéo thả co giãn trực tiếp trên hình!"
)

# 2. Khởi tạo MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Hàm tạo object đường kẻ cho Canvas
def create_fabric_line(x_center, y_center, width_px, color):
    return {
        "type": "line",
        "left": x_center,
        "top": y_center,
        "width": width_px,
        "height": 0,
        "fill": "",
        "stroke": color,
        "strokeWidth": 6,
        "x1": -width_px / 2,
        "y1": 0,
        "x2": width_px / 2,
        "y2": 0,
        "originX": "center",
        "originY": "center",
        "scaleX": 1,
        "scaleY": 1,
        "angle": 0,
        "transparentCorners": False,
        "cornerColor": "white",
        "cornerStrokeColor": "black",
        "borderColor": "black",
        "cornerSize": 18,
    }

# 3. Form nhập liệu
uploaded_file = st.file_uploader(
    "1. Chọn hoặc chụp ảnh toàn thân (JPG, PNG)", type=["jpg", "jpeg", "png"]
)
real_height_cm = st.number_input(
    "2. Nhập chiều cao thực tế của bạn (cm):",
    min_value=100.0, max_value=250.0, value=160.0, step=1.0,
)

if uploaded_file is not None:
    # Quản lý ID file ảnh để không bị reset khi kéo thả
    file_id = uploaded_file.name
    if "img_id" not in st.session_state or st.session_state.img_id != file_id:
        st.session_state.img_id = file_id
        st.session_state.initial_drawing = None

    # Load ảnh và chuẩn hóa kích thước cố định (600px width)
    image = Image.open(uploaded_file).convert("RGB")
    base_width = 600
    w_percent = base_width / float(image.size[0])
    h_size = int(float(image.size[1]) * float(w_percent))
    image = image.resize((base_width, h_size), Image.Resampling.LANCZOS)
    
    # Ép ảnh sang mảng numpy để hiển thị hoàn hảo làm background cho canvas
    bg_image_np = np.array(image)

    # Nếu upload ảnh mới, chạy AI để chấm điểm khởi tạo vị trí 3 đường kẻ
    if st.session_state.initial_drawing is None:
        results = pose.process(bg_image_np)
        
        # Giá trị mặc định phòng hờ AI không thấy người
        sx_c, sy_c, sh_w = base_width / 2, h_size * 0.25, base_width * 0.3
        wx_c, wy_c, ws_w = base_width / 2, h_size * 0.40, base_width * 0.25
        hx_c, hy_c, hp_w = base_width / 2, h_size * 0.55, base_width * 0.35
        pixel_height_val = h_size * 0.8

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            nose = lm[mp_pose.PoseLandmark.NOSE]
            l_sh, r_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER], lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            l_hip, r_hip = lm[mp_pose.PoseLandmark.LEFT_HIP], lm[mp_pose.PoseLandmark.RIGHT_HIP]
            l_ank, r_ank = lm[mp_pose.PoseLandmark.LEFT_ANKLE], lm[mp_pose.PoseLandmark.RIGHT_ANKLE]

            sx_c = (l_sh.x + r_sh.x) / 2 * base_width
            sy_c = (l_sh.y + r_sh.y) / 2 * h_size
            hx_c = (l_hip.x + r_hip.x) / 2 * base_width
            hy_c = (l_hip.y + r_hip.y) / 2 * h_size
            wx_c, wy_c = sx_c, sy_c + (hy_c - sy_c) * 0.55
            
            sh_w = abs(l_sh.x - r_sh.x) * base_width * 1.25
            hp_w = abs(l_hip.x - r_hip.x) * base_width * 1.20
            ws_w = hp_w * 0.80

            mid_ankle_y = (l_ank.y + r_ank.y) / 2 * h_size
            pixel_height_val = abs(mid_ankle_y - (nose.y * h_size)) * 1.15

        st.session_state.pixel_height = pixel_height_val
        st.session_state.initial_drawing = {
            "version": "4.4.0",
            "objects": [
                create_fabric_line(sx_c, sy_c, sh_w, "#FF0000"),  # Đỏ = Vai
                create_fabric_line(wx_c, wy_c, ws_w, "#00FF00"),  # Xanh lá = Eo
                create_fabric_line(hx_c, hy_c, hp_w, "#0000FF"),  # Xanh dương = Hông
            ]
        }

    # --- KHU VỰC VẼ TƯƠNG TÁC CÓ HÌNH NỀN ---
    st.subheader("3. Kéo thả chỉnh đường kẻ trên ảnh của bạn")
    
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=6,
        background_image=image,  # Đưa ảnh gốc vào làm background chuẩn xác
        update_streamlit=True,
        height=h_size,
        width=base_width,
        drawing_mode="transform",
        initial_drawing=st.session_state.initial_drawing,
        key="canvas_image",
    )

    # --- ĐỌC KẾT QUẢ KÉO THẢ & QUY ĐỔI RA CM ---
    cm_per_pixel = real_height_cm / st.session_state.pixel_height if st.session_state.pixel_height > 0 else 0.2
    
    final_sh = final_ws = final_hp = 30

    if canvas_result.json_data is not None and "objects" in canvas_result.json_data:
        for obj in canvas_result.json_data["objects"]:
            if obj["type"] == "line":
                length_px = math.sqrt((obj["width"] * obj["scaleX"])**2 + ((obj["height"] or 0) * obj["scaleY"])**2)
                cm_val = int(length_px * cm_per_pixel)
                
                if obj["stroke"] == "#FF0000": final_sh = max(1, cm_val)
                elif obj["stroke"] == "#00FF00": final_ws = max(1, cm_val)
                elif obj["stroke"] == "#0000FF": final_hp = max(1, cm_val)

    st.markdown("---")
    st.markdown(f"### 📊 Số đo thực tế sau khi tinh chỉnh:")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rộng Vai", f"{final_sh} cm")
    col2.metric("Rộng Eo", f"{final_ws} cm")
    col3.metric("Rộng Hông", f"{final_hp} cm")

    # --- KẾT QUẢ PHÂN TÍCH ---
    st.markdown("---")
    st.header("✨ Bước 4: Phân Tích Dáng Người & Gợi Ý Phối Đồ")

    if final_ws > final_sh and final_ws > final_hp:
        shape = "Dáng Quả Táo (Apple)"
        advice = "- **Nên mặc:** Áo cổ chữ V sâu, đầm chữ A, quần cạp cao.\n- **Tránh mặc:** Áo bó sát vòng 2."
    elif final_hp > final_sh * 1.05:
        shape = "Dáng Quả Lê (Pear)"
        advice = "- **Nên mặc:** Áo trễ vai, bèo nhún phần ngực, quần ống suông/tối màu.\n- **Tránh mặc:** Quần skinny sáng màu."
    elif final_sh > final_hp * 1.05:
        shape = "Dáng Tam Giác Ngược"
        advice = "- **Nên mặc:** Váy chữ A, váy xòe, quần ống rộng.\n- **Tránh mặc:** Áo độn vai, áo trễ vai."
    elif abs(final_sh - final_hp) < (final_sh * 0.08) and final_ws < (final_hp * 0.80):
        shape = "Dáng Đồng Hồ Cát (Hourglass)"
        advice = "- **Nên mặc:** Đầm ôm body, áo croptop, quần cạp cao tôn eo.\n- **Tránh mặc:** Quần áo oversize rộng thùng thình."
    else:
        shape = "Dáng Chữ Nhật (Rectangle)"
        advice = "- **Nên mặc:** Áo có điểm nhấn ở eo, váy xòe bồng bềnh.\n- **Tránh mặc:** Trang phục suông tuột."

    st.subheader(f"📍 Vóc dáng của bạn: **{shape}**")
    st.markdown(advice)
