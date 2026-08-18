import math
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# 1. Cấu hình giao diện
st.set_page_config(page_title="AI Body Scanner Ultra", page_icon="👗", layout="centered")
st.title("👗 AI Quét Dáng Viền Cơ Thể (Ultra)")
st.markdown("---")
st.write(
    "💡 **Mẹo chụp:** Hãy đứng thẳng, **dang nhẹ hai tay (chữ A)** để AI nhìn rõ phần eo. "
    "Thuật toán mới sẽ tự động dò tìm điểm ngoài cùng của Vai, điểm hẹp nhất của Eo và điểm to nhất của Hông dựa trên viền cơ thể."
)

# 2. Khởi tạo MediaPipe Pose & Segmentation (Tách nền)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

mp_segmentation = mp.solutions.selfie_segmentation
segmentation = mp_segmentation.SelfieSegmentation(model_selection=1)

# Hàm tạo object đường kẻ cho Canvas
def create_fabric_line(x_center, y_center, width_px, color):
    return {
        "type": "line", "left": x_center, "top": y_center, "width": width_px,
        "height": 0, "fill": "", "stroke": color, "strokeWidth": 6,
        "x1": -width_px / 2, "y1": 0, "x2": width_px / 2, "y2": 0,
        "originX": "center", "originY": "center", "scaleX": 1, "scaleY": 1,
        "angle": 0, "transparentCorners": False, "cornerColor": "white",
        "cornerStrokeColor": "black", "borderColor": "black", "cornerSize": 18,
    }

# Hàm dò tìm viền cơ thể (từ tâm xương sống đi ra 2 bên)
def get_contour_width(mask, y_start, y_end, center_x, mode="max"):
    best_y = int((y_start + y_end) / 2)
    best_left, best_right = int(center_x), int(center_x)
    best_w = -1 if mode == "max" else float('inf')
    
    for y in range(int(y_start), int(y_end)):
        if y < 0 or y >= mask.shape[0]: continue
        row = mask[y, :]
        c_x = int(center_x)
        if c_x < 0 or c_x >= len(row) or not row[c_x]: 
            continue # Bỏ qua nếu tâm bị sai
            
        # Dò sang trái và phải cho đến khi gặp viền (pixel đen)
        l = c_x
        while l > 0 and row[l]: l -= 1
        r = c_x
        while r < len(row)-1 and row[r]: r += 1
        
        w = r - l
        if mode == "max" and w > best_w:
            best_w = w; best_left = l; best_right = r; best_y = y
        elif mode == "min" and w < best_w:
            best_w = w; best_left = l; best_right = r; best_y = y
            
    if best_w == -1 or best_w == float('inf'):
        return None
    return best_y, best_left, best_right, best_w

# 3. Form nhập liệu
uploaded_file = st.file_uploader(
    "1. Chọn hoặc chụp ảnh toàn thân (JPG, PNG)", type=["jpg", "jpeg", "png"]
)
real_height_cm = st.number_input(
    "2. Nhập chiều cao thực tế của bạn (cm):",
    min_value=100.0, max_value=250.0, value=160.0, step=1.0,
)

if uploaded_file is not None:
    file_id = uploaded_file.name
    if "img_id" not in st.session_state or st.session_state.img_id != file_id:
        st.session_state.img_id = file_id
        st.session_state.initial_drawing = None

    # Load ảnh và chuẩn hóa
    image = Image.open(uploaded_file).convert("RGB")
    base_width = 600
    w_percent = base_width / float(image.size[0])
    h_size = int(float(image.size[1]) * float(w_percent))
    image = image.resize((base_width, h_size), Image.Resampling.LANCZOS)
    bg_image_np = np.array(image)

    if st.session_state.initial_drawing is None:
        # Chạy Pose và Segmentation
        pose_results = pose.process(bg_image_np)
        seg_results = segmentation.process(bg_image_np)
        
        # Mặc định
        sx_c, sy_c, sh_w = base_width / 2, h_size * 0.25, base_width * 0.3
        wx_c, wy_c, ws_w = base_width / 2, h_size * 0.40, base_width * 0.25
        hx_c, hy_c, hp_w = base_width / 2, h_size * 0.55, base_width * 0.35
        st.session_state.pixel_height = h_size * 0.8

        if pose_results.pose_landmarks and seg_results.segmentation_mask is not None:
            # Tạo mask nhị phân (phần trắng là cơ thể, đen là nền)
            mask = seg_results.segmentation_mask > 0.5
            
            lm = pose_results.pose_landmarks.landmark
            nose = lm[mp_pose.PoseLandmark.NOSE]
            l_sh, r_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER], lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            l_hip, r_hip = lm[mp_pose.PoseLandmark.LEFT_HIP], lm[mp_pose.PoseLandmark.RIGHT_HIP]
            l_ank, r_ank = lm[mp_pose.PoseLandmark.LEFT_ANKLE], lm[mp_pose.PoseLandmark.RIGHT_ANKLE]

            # Xác định Y và tâm X từ khung xương
            y_sh_avg = (l_sh.y + r_sh.y) / 2 * h_size
            y_hip_avg = (l_hip.y + r_hip.y) / 2 * h_size
            center_x = (l_sh.x + r_sh.x) / 2 * base_width

            # 1. Tìm VAI (Khoảng lớn nhất quanh vùng vai)
            sh_data = get_contour_width(mask, y_sh_avg - h_size*0.05, y_sh_avg + h_size*0.05, center_x, "max")
            if sh_data:
                sy_c = sh_data[0]
                sx_c = (sh_data[1] + sh_data[2]) / 2
                sh_w = sh_data[3]

            # 2. Tìm EO (Khoảng hẹp nhất giữa vai và hông)
            ws_data = get_contour_width(mask, y_sh_avg + h_size*0.1, y_hip_avg - h_size*0.05, center_x, "min")
            if ws_data:
                wy_c = ws_data[0]
                wx_c = (ws_data[1] + ws_data[2]) / 2
                ws_w = ws_data[3]

            # 3. Tìm HÔNG (Khoảng lớn nhất từ hông trở xuống)
            hp_data = get_contour_width(mask, y_hip_avg - h_size*0.05, y_hip_avg + h_size*0.1, center_x, "max")
            if hp_data:
                hy_c = hp_data[0]
                hx_c = (hp_data[1] + hp_data[2]) / 2
                hp_w = hp_data[3]

            # Tính chiều cao
            mid_ankle_y = (l_ank.y + r_ank.y) / 2 * h_size
            st.session_state.pixel_height = abs(mid_ankle_y - (nose.y * h_size)) * 1.15

        st.session_state.initial_drawing = {
            "version": "4.4.0",
            "objects": [
                create_fabric_line(sx_c, sy_c, sh_w, "#FF0000"),  # Vai
                create_fabric_line(wx_c, wy_c, ws_w, "#00FF00"),  # Eo
                create_fabric_line(hx_c, hy_c, hp_w, "#0000FF"),  # Hông
            ]
        }

    # --- KHU VỰC VẼ TƯƠNG TÁC ---
    st.subheader("3. Kéo thả chỉnh đường kẻ (nếu cần)")
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=6,
        background_image=image,
        update_streamlit=True,
        height=h_size,
        width=base_width,
        drawing_mode="transform",
        initial_drawing=st.session_state.initial_drawing,
        key="canvas_img",
    )

    # --- TÍNH TOÁN KẾT QUẢ ---
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
    st.markdown("### 📊 Số đo thực tế theo tỷ lệ ảnh:")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rộng Vai", f"{final_sh} cm")
    col2.metric("Rộng Eo", f"{final_ws} cm")
    col3.metric("Rộng Hông", f"{final_hp} cm")

    # --- PHÂN TÍCH DÁNG NGƯỜI ---
    st.markdown("---")
    st.header("✨ Bước 4: Phân Tích Dáng Người")

    if final_ws > final_sh and final_ws > final_hp:
        shape = "Dáng Quả Táo (Apple)"
    elif final_hp > final_sh * 1.05:
        shape = "Dáng Quả Lê (Pear)"
    elif final_sh > final_hp * 1.05:
        shape = "Dáng Tam Giác Ngược"
    elif abs(final_sh - final_hp) < (final_sh * 0.08) and final_ws < (final_hp * 0.80):
        shape = "Dáng Đồng Hồ Cát (Hourglass)"
    else:
        shape = "Dáng Chữ Nhật (Rectangle)"

    st.subheader(f"📍 Dáng của bạn: **{shape}**")
