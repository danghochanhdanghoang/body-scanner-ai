import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

# ==========================================
# 1. HÀM PHÂN LOẠI DÁNG NGƯỜI
# ==========================================
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

# ==========================================
# 2. CƠ SỞ DỮ LIỆU TỦ ĐỒ (ĐÃ ĐIỀN VÀO ĐÂY)
# ==========================================
WARDROBE_DB = {
    "Hourglass (Dáng Đồng Hồ Cát)": {
        "👕 Áo / Tops": [
            {"name": "Áo thun/cổ vuông ôm sát", "desc": "Tôn đường thắt eo tự nhiên và phần vai cân đối."},
            {"name": "Áo sơ mi chiết eo / Wrap top", "desc": "Tạo điểm nhấn tối đa vào vòng 2."}
        ],
        "👖 Quần & Chân váy": [
            {"name": "Quần Jean/Tây cạp cao", "desc": "Kéo dài chân và ôm trọn phần hông chuẩn."},
            {"name": "Chân váy bút chì / Váy đuôi cá", "desc": "Khoe trọn đường cong từ eo xuống hông."}
        ],
        "👗 Đầm liền": [
            {"name": "Đầm Wrap Dress (Đầm quấn)", "desc": "Thiết kế kinh điển hoàn hảo nhất cho dáng đồng hồ cát."},
            {"name": "Đầm Bodycon ôm sát", "desc": "Tôn trọn vẹn 3 vòng."}
        ]
    },
    "Pear (Dáng Quả Lê)": {
        "👕 Áo / Tops": [
            {"name": "Áo trễ vai / Có bèo nhún vai", "desc": "Tạo cảm giác vai rộng hơn để cân bằng với hông."},
            {"name": "Áo có họa tiết nổi bật/màu sáng", "desc": "Thu hút ánh nhìn vào phần thân trên."}
        ],
        "👖 Quần & Chân váy": [
            {"name": "Quần ống suông/ống rộng tối màu", "desc": "Che khuyết điểm hông đùi to, tạo nét thanh thoát."},
            {"name": "Chân váy chữ A (A-line)", "desc": "Xòe nhẹ từ eo xuống, che hông nở cực tốt."}
        ],
        "👗 Đầm liền": [
            {"name": "Đầm xòe chữ A nhấn eo", "desc": "Tập trung vào eo nhỏ và che đi phần hông."}
        ]
    },
    "Inverted Triangle (Tam Giác Ngược)": {
        "👕 Áo / Tops": [
            {"name": "Áo cổ chữ V sâu / Cổ tim", "desc": "Thu hẹp phần vai rộng, tạo cảm giác thân trên thon gọn."},
            {"name": "Áo tay Raglan / Áo tối màu", "desc": "Giảm độ chú ý vào khung vai."}
        ],
        "👖 Quần & Chân váy": [
            {"name": "Quần Cargo / Quần ống rộng cạp cao", "desc": "Tạo độ phồng phần dưới để cân bằng với vai."},
            {"name": "Chân váy xếp ly / Váy xòe tròn", "desc": "Tăng kích thước thị giác cho hông."}
        ],
        "👗 Đầm liền": [
            {"name": "Đầm Peplum / Đầm hạ eo", "desc": "Tạo thêm độ nẩy cho hông."}
        ]
    },
    "Rectangle (Dáng Chữ Nhật)": {
        "👕 Áo / Tops": [
            {"name": "Áo Croptop / Áo cổ đổ", "desc": "Tạo đường cong giả cho phần thân trên."},
            {"name": "Áo nhún eo / Có thắt lưng", "desc": "Tạo cảm giác có thắt eo."}
        ],
        "👖 Quần & Chân váy": [
            {"name": "Chân váy tầng / Váy xòe phồng", "desc": "Tạo độ nẩy cho hông."},
            {"name": "Quần Baggy / Quần tây xếp li", "desc": "Giúp phần mông và hông đầy đặn hơn."}
        ],
        "👗 Đầm liền": [
            {"name": "Đầm thắt thắt lưng / Đầm xòe công chúa", "desc": "Chia lại tỷ lệ cơ thể rõ ràng."}
        ]
    },
    "Apple (Dáng Quả Táo)": {
        "👕 Áo / Tops": [
            {"name": "Áo cổ chữ V dáng rủ", "desc": "Kéo dài phần cổ và ngực, giấu bụng."},
            {"name": "Áo khoác Blazer dáng dài (mở cúc)", "desc": "Tạo 2 đường dọc kéo dài cơ thể."}
        ],
        "👖 Quần & Chân váy": [
            {"name": "Quần ống đứng / Quần Bootcut", "desc": "Khoe đôi chân thon gọn đặc trưng."},
            {"name": "Chân váy chữ A cạp vừa", "desc": "Giữ phần bụng thoải mái mà vẫn xòe nhẹ."}
        ],
        "👗 Đầm liền": [
            {"name": "Đầm Suông / Đầm Empire", "desc": "Giấu hoàn toàn vòng eo đầy đặn."}
        ]
    }
}

# ==========================================
# 3. HÀM XỬ LÝ ẢNH CHÍNH
# ==========================================
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

        def get_body_width_from_center(y_coord, center_x):
            if y_coord < 0 or y_coord >= h: return 0, 0, 0
            row = mask[y_coord, :]
            cx = int(center_x)
            if cx < 0 or cx >= w or row[cx] == 0: return 0, 0, 0
            
            x1 = cx
            while x1 > 0 and row[x1] > 0: x1 -= 1
            x2 = cx
            while x2 < w - 1 and row[x2] > 0: x2 += 1
            return (x2 - x1), x1, x2

        y_shoulder_avg = int((l_shoulder[1] + r_shoulder[1]) / 2)
        y_hip_avg = int((l_hip[1] + r_hip[1]) / 2)
        x_sh_center = (l_shoulder[0] + r_shoulder[0]) / 2
        x_hip_center = (l_hip[0] + r_hip[0]) / 2

        # 1. Đo Vai
        sh_distance_px = int(np.linalg.norm(l_shoulder - r_shoulder))
        shoulder_width_px = int(sh_distance_px * 1.20) 
        shoulder_cm = round(shoulder_width_px * px_to_cm, 1)
        sh_y = y_shoulder_avg
        sh_x1 = int(x_sh_center - (shoulder_width_px / 2))
        sh_x2 = int(x_sh_center + (shoulder_width_px / 2))

        # 2. Đo Eo
        min_waist_px, waist_y, waist_x1, waist_x2 = float('inf'), -1, 0, 0
        for y in range(y_shoulder_avg + int((y_hip_avg - y_shoulder_avg) * 0.30), y_shoulder_avg + int((y_hip_avg - y_shoulder_avg) * 0.80)):
            progress = (y - y_shoulder_avg) / (y_hip_avg - y_shoulder_avg + 1e-6)
            cx = x_sh_center + (x_hip_center - x_sh_center) * progress
            width, x1, x2 = get_body_width_from_center(y, cx)
            if 0 < width < min_waist_px: min_waist_px, waist_y, waist_x1, waist_x2 = width, y, x1, x2
        if min_waist_px == float('inf'): min_waist_px = 0
        waist_cm = round(min_waist_px * px_to_cm, 1)

        # 3. Đo Hông
        max_hip_px, hip_y, hip_x1, hip_x2 = 0, -1, 0, 0
        for y in range(y_shoulder_avg + int((y_hip_avg - y_shoulder_avg) * 0.80), min(h - 1, y_shoulder_avg + int((y_hip_avg - y_shoulder_avg) * 1.25))):
            width, x1, x2 = get_body_width_from_center(y, x_hip_center)
            if width > max_hip_px: max_hip_px, hip_y, hip_x1, hip_x2 = width, y, x1, x2
        hip_cm = round(max_hip_px * px_to_cm, 1)

        shape_name, advice = classify_body_shape(shoulder_cm, waist_cm, hip_cm)
        annotated_img = image_np.copy()
        
        cv2.line(annotated_img, (sh_x1, sh_y), (sh_x2, sh_y), (0, 255, 0), 4)
        if waist_y != -1: cv2.line(annotated_img, (waist_x1, waist_y), (waist_x2, waist_y), (255, 0, 0), 4)
        if hip_y != -1: cv2.line(annotated_img, (hip_x1, hip_y), (hip_x2, hip_y), (0, 0, 255), 4)

        result_data = {
            "shoulder": shoulder_cm, "waist": waist_cm, "hip": hip_cm,
            "shape": shape_name, "advice": advice
        }
        return annotated_img, result_data

# ==========================================
# 4. GIAO DIỆN WEB VỚI STREAMLIT & TỦ ĐỒ
# ==========================================
st.set_page_config(page_title="AI Stylist - Đo tỷ lệ cơ thể", layout="centered")

st.title("👗 AI Stylist - Phân tích dáng người")
st.write("Upload ảnh toàn thân hoặc chụp trực tiếp để AI tư vấn cách phối đồ cho bạn!")

user_height = st.number_input("Nhập chiều cao của bạn (cm):", min_value=100.0, max_value=250.0, value=165.0, step=1.0)
uploaded_file = st.file_uploader("Chọn ảnh hoặc chụp ảnh mới", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    
    st.image(image, caption="Ảnh gốc", use_column_width=True)
    
    if st.button("Phân tích ngay", type="primary"):
        with st.spinner("AI đang quét tỷ lệ cơ thể..."):
            annotated_img, result = process_image(image_np, user_height)
            
            if isinstance(result, str):
                st.error(result)
            else:
                st.success("Phân tích thành công!")
                st.image(annotated_img, caption="Ảnh AI đã quét", use_column_width=True)
                
                st.subheader("📊 Kết quả đo lường")
                col1, col2, col3 = st.columns(3)
                col1.metric("Vai", f"{result['shoulder']} cm")
                col2.metric("Eo", f"{result['waist']} cm")
                col3.metric("Hông", f"{result['hip']} cm")
                
                st.subheader("✨ Dáng người của bạn")
                st.info(f"**{result['shape']}**")
                
                # HIỂN THỊ TỦ ĐỒ THEO TABS VÀO ĐÂY
                st.subheader("🛍️ Tủ đồ gợi ý riêng cho bạn")
                shape_key = result['shape']
                
                if shape_key in WARDROBE_DB:
                    wardrobe = WARDROBE_DB[shape_key]
                    tab1, tab2, tab3 = st.tabs(["👕 Áo đề xuất", "👖 Quần & Chân váy", "👗 Đầm liền"])
                    
                    with tab1:
                        for item in wardrobe["👕 Áo / Tops"]:
                            st.success(f"**{item['name']}**")
                            st.caption(f"💡 *Lý do chọn:* {item['desc']}")
                    with tab2:
                        for item in wardrobe["👖 Quần & Chân váy"]:
                            st.info(f"**{item['name']}**")
                            st.caption(f"💡 *Lý do chọn:* {item['desc']}")
                    with tab3:
                        for item in wardrobe["👗 Đầm liền"]:
                            st.warning(f"**{item['name']}**")
                            st.caption(f"💡 *Lý do chọn:* {item['desc']}")
