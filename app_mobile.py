import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
from rembg import remove

st.set_page_config(page_title="AI Stylist & Phòng Thử Đồ", layout="wide")

# Bộ nhớ lưu Tủ đồ trong Session
if "user_tops" not in st.session_state:
    st.session_state["user_tops"] = []
if "user_bottoms" not in st.session_state:
    st.session_state["user_bottoms"] = []

# ==========================================
# 1. HÀM PHÂN LOẠI DÁNG NGƯỜI
# ==========================================
def classify_body_shape(shoulder_cm, waist_cm, hip_cm):
    if shoulder_cm <= 0 or waist_cm <= 0 or hip_cm <= 0:
        return "Uncertain", "Không đủ dữ liệu."
    
    if (waist_cm / shoulder_cm >= 0.85) or (waist_cm / hip_cm >= 0.85):
        return "Apple (Dáng Quả Táo)", "Phần thân trên và eo đầy đặn. Nên chọn áo cổ V, đầm dáng xòe nhẹ."
    elif (shoulder_cm / hip_cm) >= 1.05:
        return "Inverted Triangle (Tam Giác Ngược)", "Vai rộng, hông hẹp. Nên mặc quần ống rộng, chân váy xếp ly."
    elif (hip_cm / shoulder_cm) >= 1.05:
        return "Pear (Dáng Quả Lê)", "Hông nở, vai nhỏ. Nên chọn áo bèo nhún/trễ vai, quần suông tối màu."
    else:
        if (waist_cm / min(shoulder_cm, hip_cm)) <= 0.75:
            return "Hourglass (Dáng Đồng Hồ Cát)", "Tỷ lệ chuẩn thắt eo rõ. Hãy chọn đầm ôm sát, áo chiết eo."
        else:
            return "Rectangle (Dáng Chữ Nhật)", "Thân hình thẳng. Nên tạo điểm nhấn bằng thắt lưng, chân váy xòe."

# ==========================================
# 2. HÀM TÁCH NỀN QUẦN ÁO
# ==========================================
def process_clothing_item(img_pil):
    return remove(img_pil)

# ==========================================
# 3. HÀM QUÉT AI & ĐO CƠ THỂ (GIỮ NGUYÊN VẼ VẠCH)
# ==========================================
def analyze_full_body(image_np, user_height_cm):
    mp_pose = mp.solutions.pose
    h, w, _ = image_np.shape

    with mp_pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=True, min_detection_confidence=0.6) as pose:
        results = pose.process(image_np)
        if not results.pose_landmarks:
            return None, None, "Không tìm thấy dáng người trong ảnh!"

        mask = (results.segmentation_mask > 0.5).astype(np.uint8) * 255
        y_indices = np.where(mask > 0)[0]
        if len(y_indices) == 0:
            return None, None, "Không trích xuất được phom dáng."
            
        total_height_px = np.max(y_indices) - np.min(y_indices)
        px_to_cm = user_height_cm / total_height_px if total_height_px > 0 else 0

        landmarks = results.pose_landmarks.landmark
        l_sh = np.array([int(landmarks[11].x * w), int(landmarks[11].y * h)])
        r_sh = np.array([int(landmarks[12].x * w), int(landmarks[12].y * h)])
        l_hip = np.array([int(landmarks[23].x * w), int(landmarks[23].y * h)])
        r_hip = np.array([int(landmarks[24].x * w), int(landmarks[24].y * h)])

        def get_body_width_from_center(y_coord, center_x):
            if y_coord < 0 or y_coord >= h: return 0, 0, 0
            row = mask[y_coord, :]
            cx = int(center_x)
            if cx < 0 or cx >= w or row[cx] == 0: return 0, 0, 0
            x1, x2 = cx, cx
            while x1 > 0 and row[x1] > 0: x1 -= 1
            while x2 < w - 1 and row[x2] > 0: x2 += 1
            return (x2 - x1), x1, x2

        y_sh_avg = int((l_sh[1] + r_sh[1]) / 2)
        y_hip_avg = int((l_hip[1] + r_hip[1]) / 2)
        x_sh_center = (l_sh[0] + r_sh[0]) / 2
        x_hip_center = (l_hip[0] + r_hip[0]) / 2

        # Đo Vai
        sh_distance_px = int(np.linalg.norm(l_sh - r_sh))
        shoulder_width_px = int(sh_distance_px * 1.20)
        shoulder_cm = round(shoulder_width_px * px_to_cm, 1)
        sh_x1 = int(x_sh_center - (shoulder_width_px / 2))
        sh_x2 = int(x_sh_center + (shoulder_width_px / 2))

        # Đo Eo
        min_waist_px, waist_y, w_x1, w_x2 = float('inf'), -1, 0, 0
        for y in range(y_sh_avg + int((y_hip_avg - y_sh_avg) * 0.30), y_sh_avg + int((y_hip_avg - y_sh_avg) * 0.80)):
            progress = (y - y_sh_avg) / (y_hip_avg - y_sh_avg + 1e-6)
            cx = x_sh_center + (x_hip_center - x_sh_center) * progress
            width, x1, x2 = get_body_width_from_center(y, cx)
            if 0 < width < min_waist_px: min_waist_px, waist_y, w_x1, w_x2 = width, y, x1, x2
        if min_waist_px == float('inf'): min_waist_px = 0
        waist_cm = round(min_waist_px * px_to_cm, 1)

        # Đo Hông
        max_hip_px, hip_y, h_x1, h_x2 = 0, -1, 0, 0
        for y in range(y_sh_avg + int((y_hip_avg - y_sh_avg) * 0.80), min(h - 1, y_sh_avg + int((y_hip_avg - y_sh_avg) * 1.25))):
            width, x1, x2 = get_body_width_from_center(y, x_hip_center)
            if width > max_hip_px: max_hip_px, hip_y, h_x1, h_x2 = width, y, x1, x2
        hip_cm = round(max_hip_px * px_to_cm, 1)

        shape_name, advice = classify_body_shape(shoulder_cm, waist_cm, hip_cm)

        # Vẽ 3 đường kẻ màu (Xanh lá - Vai, Xanh dương - Eo, Đỏ - Hông)
        annotated_img = image_np.copy()
        cv2.line(annotated_img, (sh_x1, y_sh_avg), (sh_x2, y_sh_avg), (0, 255, 0), 4)
        if waist_y != -1: cv2.line(annotated_img, (w_x1, waist_y), (w_x2, waist_y), (255, 0, 0), 4)
        if hip_y != -1: cv2.line(annotated_img, (h_x1, hip_y), (h_x2, hip_y), (0, 0, 255), 4)

        result_data = {
            "shoulder": shoulder_cm, "waist": waist_cm, "hip": hip_cm,
            "shape": shape_name, "advice": advice,
            "sh_center": (int(x_sh_center), y_sh_avg),
            "hip_center": (int(x_hip_center), y_hip_avg),
            "sh_width_px": shoulder_width_px,
            "hip_width_px": max_hip_px if max_hip_px > 0 else int(np.linalg.norm(l_hip - r_hip) * 1.2)
        }
        return annotated_img, result_data, "OK"

# ==========================================
# 4. HÀM MẶC THỬ ĐỒ TRÊN KHUNG XƯƠNG
# ==========================================
def overlay_clothing(body_pil, pose_data, top_pil=None, bottom_pil=None):
    base_img = body_pil.convert("RGBA")
    
    if top_pil is not None:
        top_img = top_pil.convert("RGBA")
        target_w = pose_data["sh_width_px"]
        target_h = int(target_w * (top_img.height / top_img.width))
        resized_top = top_img.resize((target_w, target_h))
        pos_x = pose_data["sh_center"][0] - (target_w // 2)
        pos_y = pose_data["sh_center"][1] - int(target_h * 0.15)
        base_img.paste(resized_top, (pos_x, pos_y), resized_top)

    if bottom_pil is not None:
        bot_img = bottom_pil.convert("RGBA")
        target_w = pose_data["hip_width_px"]
        target_h = int(target_w * (bot_img.height / bot_img.width))
        resized_bot = bot_img.resize((target_w, target_h))
        pos_x = pose_data["hip_center"][0] - (target_w // 2)
        pos_y = pose_data["hip_center"][1] - int(target_h * 0.10)
        base_img.paste(resized_bot, (pos_x, pos_y), resized_bot)

    return base_img

# ==========================================
# 5. GIAO DIỆN WEB STREAMLIT
# ==========================================
st.title("👗 AI Stylist: Phân Tích Dáng & Phòng Thử Đồ Ảo")

tab1, tab2, tab3 = st.tabs(["📐 1. Quét Dáng & Số Đo AI", "📸 2. Tủ Đồ (Thêm & Tách Nền)", "🪞 3. Phòng Thử Đồ Ảo"])

# TAB 1: ĐO SỐ ĐO VÀ CHẨN ĐOÁN DÁNG NGƯỜI
with tab1:
    st.subheader("Chụp/Upload ảnh toàn thân để AI đo 3 vòng")
    u_height = st.number_input("Chiều cao của bạn (cm):", 100.0, 250.0, 165.0, 1.0)
    body_file = st.file_uploader("Tải ảnh toàn thân đứng thẳng", type=["jpg", "png", "jpeg"], key="body_scan")

    if body_file and st.button("Phân Tích Dáng Ngay", type="primary"):
        with st.spinner("AI đang tính toán chỉ số & quét phom dáng..."):
            img_pil = Image.open(body_file)
            img_np = np.array(img_pil)
            ann_img, res, msg = analyze_full_body(img_np, u_height)

            if res is None:
                st.error(msg)
            else:
                st.session_state["scanned_body_pil"] = img_pil
                st.session_state["scanned_pose_data"] = res

                col_a, col_b = st.columns([1, 1])
                with col_a:
                    st.image(ann_img, caption="Kết quả quét tỷ lệ khung xương", use_column_width=True)
                with col_b:
                    st.subheader("📊 Số đo ước tính")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Vai", f"{res['shoulder']} cm")
                    c2.metric("Eo", f"{res['waist']} cm")
                    c3.metric("Hông", f"{res['hip']} cm")

                    st.subheader("✨ Dáng người của bạn")
                    st.info(f"**{res['shape']}**")
                    st.write(f"💡 **Tư vấn stylist:** {res['advice']}")

# TAB 2: THÊM ĐỒ VÀO TỦ & TÁCH NỀN TỰ ĐỘNG
with tab2:
    st.subheader("Tải ảnh quần/áo thực tế của bạn lên")
    c_type = st.radio("Loại trang phục:", ["Áo (Tops)", "Quần / Chân váy (Bottoms)"], horizontal=True)
    c_file = st.file_uploader("Chụp/Upload ảnh trang phục", type=["jpg", "png", "jpeg"], key="cloth_upload")

    if c_file and st.button("Tách nền & Lưu tủ đồ"):
        with st.spinner("AI đang loại bỏ phông nền..."):
            raw_c = Image.open(c_file)
            clean_c = process_clothing_item(raw_c)
            if "Áo" in c_type:
                st.session_state["user_tops"].append(clean_c)
                st.success("Đã thêm Áo vào Tủ đồ!")
            else:
                st.session_state["user_bottoms"].append(clean_c)
                st.success("Đã thêm Quần/Váy vào Tủ đồ!")

# TAB 3: PHÒNG THỬ ĐỒ ẢO (MẶC ĐỒ LÊN NGUỜI)
with tab3:
    st.subheader("Mặc thử quần áo thực tế lên phom dáng của bạn")
    if "scanned_body_pil" not in st.session_state:
        st.warning("⚠️ Bạn cần sang **Tab 1** tải ảnh và bấm 'Phân Tích Dáng Ngay' trước!")
    else:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.write("**1. Chọn Áo từ tủ:**")
            sel_top = None
            if len(st.session_state["user_tops"]) > 0:
                t_idx = st.selectbox("Danh sách Áo:", range(len(st.session_state["user_tops"])), format_func=lambda x: f"Áo {x+1}")
                sel_top = st.session_state["user_tops"][t_idx]
                st.image(sel_top, width=100)
            else:
                st.info("Chưa có áo nào. Hãy sang Tab 2 tải áo lên!")

            st.write("**2. Chọn Quần/Váy từ tủ:**")
            sel_bot = None
            if len(st.session_state["user_bottoms"]) > 0:
                b_idx = st.selectbox("Danh sách Quần/Váy:", range(len(st.session_state["user_bottoms"])), format_func=lambda x: f"Món {x+1}")
                sel_bot = st.session_state["user_bottoms"][b_idx]
                st.image(sel_bot, width=100)
            else:
                st.info("Chưa có quần/váy nào.")

        with col2:
            st.write("**🖼️ Kết quả Mặc thử (Virtual Try-On):**")
            fitted = overlay_clothing(
                st.session_state["scanned_body_pil"],
                st.session_state["scanned_pose_data"],
                sel_top,
                sel_bot
            )
            st.image(fitted, caption="Phòng thử đồ ảo", use_column_width=True)
