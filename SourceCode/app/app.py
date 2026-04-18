import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import sys
import os
import sqlite3
import pandas as pd
from datetime import datetime
import plotly.express as px

# Cấu hình Streamlit 
st.set_page_config(page_title="GreenAI - Phân loại rác", page_icon="♻️", layout="wide")

# Cấu hình Database & Hàm tiện ích 
DB_NAME = "trash_history.db"

def init_db():
    """Khởi tạo SQLite DB và tạo bảng history nếu chưa tồn tại."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_time TEXT,
            trash_type TEXT,
            confidence REAL,
            category TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_to_db(trash_type, confidence):
    """Lưu kết quả dự đoán vào Database."""
    if trash_type in ['cardboard', 'glass', 'metal', 'paper', 'plastic']:
        category = "Tái Chế"
    elif trash_type in ['organic']:
        category = "Hữu Cơ"
    elif trash_type in ['battery']:
        category = "Nguy Hại"
    else:
        category = "Vô Cơ"
    
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('INSERT INTO history (scan_time, trash_type, confidence, category) VALUES (?, ?, ?, ?)',
              (now, trash_type, confidence, category))
    conn.commit()
    conn.close()

# Tự động khởi tạo database khi khởi động web
init_db()


# Cấu hình Model & Dữ liệu
CLASS_NAMES = [
    'battery', 'cardboard', 'clothes', 'glass', 'metal', 'organic', 'paper', 'plastic', 'shoes', 'trash'
]

ADVICE = {
    'battery': "**PIN – RÁC NGUY HẠI**\n\n📌 **Cách xử lý:**\n- Không vứt chung rác thông thường.\n- Bọc kín hai đầu cực bằng băng keo.\n- Mang đến điểm thu gom pin, ắc quy gần nhất.\n\n**Thùng rác:** Thùng đặc biệt (rác nguy hại).",
    'cardboard': "**BÌA CARTON – RÁC TÁI CHẾ**\n\n📌 **Cách xử lý:**\n- Tháo băng keo, gỡ kim ghim.\n- Bóp dẹp hoặc xếp chồng để tiết kiệm diện tích.\n- Giữ khô ráo, không bị ướt hoặc dính dầu mỡ.\n\n**Thùng rác:** Màu xanh (tái chế).",
    'clothes': "**QUẦN ÁO CŨ – RÁC VÔ CƠ**\n\n📌 **Cách xử lý:**\n- Nếu còn tốt: ủng hộ hoặc bán đồ second-hand.\n- Nếu hỏng nặng: cắt nhỏ, bỏ vào túi kín.\n- Không trộn với rác hữu cơ.\n\n**Thùng rác:** Màu cam (vô cơ).",
    'glass': "**THỦY TINH – RÁC TÁI CHẾ**\n\n📌 **Cách xử lý:**\n- Đổ hết chất lỏng bên trong.\n- Rửa sạch và để ráo nước.\n- Dùng giấy báo bọc ngoài nếu chai vỡ để tránh nguy hiểm.\n\n**Thùng rác:** Màu xanh (tái chế).",
    'metal': "**KIM LOẠI – RÁC TÁI CHẾ**\n\n📌 **Cách xử lý:**\n- Đổ hết thức ăn/nước bên trong.\n- Rửa sạch và bóp dẹp lon/hộp.\n- Tháo nắp riêng (nếu là nắp nhựa).\n\n**Thùng rác:** Màu xanh (tái chế).",
    'organic': "**RÁC HỮU CƠ**\n\n📌 **Cách xử lý:**\n- Bỏ trực tiếp vào thùng, không cần túi nilon.\n- Có thể ủ phân compost tại nhà nếu có điều kiện.\n- Không lẫn tạp chất như nhựa, thủy tinh.\n\n**Thùng rác:** Màu nâu (hữu cơ).",
    'paper': "**GIẤY – RÁC TÁI CHẾ**\n\n📌 **Cách xử lý:**\n- Giấy sạch, không dính dầu mỡ hoặc thức ăn.\n- Gấp gọn hoặc cắt nhỏ.\n- Giấy bẩn (dùng lau, thấm dầu) bỏ vào rác vô cơ.\n\n**Thùng rác:** Màu xanh (tái chế).",
    'plastic': "**NHỰA – RÁC TÁI CHẾ**\n\n📌 **Cách xử lý:**\n- Đổ cạn nước, rửa sạch.\n- Bóp dẹp chai/hộp để tiết kiệm không gian.\n- Bỏ nắp riêng (nắp nhựa cũng tái chế được).\n\n**Thùng rác:** Màu xanh (tái chế).",
    'shoes': "**GIÀY DÉP CŨ – RÁC VÔ CƠ**\n\n📌 **Cách xử lý:**\n- Buộc cặp đôi lại với nhau.\n- Nếu còn dùng được: tặng cho người có nhu cầu.\n- Nếu hỏng: bỏ vào thùng rác vô cơ.\n\n**Thùng rác:** Màu cam (vô cơ).",
    'trash': "**RÁC HỖN HỢP – VÔ CƠ**\n\n📌 **Cách xử lý:**\n- Buộc kín miệng túi rác.\n- Không lẫn kim loại, thủy tinh, pin hay rác hữu cơ.\n- Đây là rác sẽ được chôn lấp hoặc đốt.\n\n**Thùng rác:** Màu cam (vô cơ)."
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@st.cache_resource
def load_model():
    """Tải mô hình EfficientNet-B0 đã được huấn luyện."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    finetune_dir = os.path.abspath(os.path.join(base_dir, '..', 'fine_tune_efficientnet'))
    
    if finetune_dir not in sys.path:
        sys.path.insert(0, finetune_dir)
        
    try:
        from efficientnet_finetune import build_efficientnet_finetune
        model = build_efficientnet_finetune(num_classes=10, unfreeze_backbone=True, unfreeze_layers=0)
        model_path = os.path.abspath(os.path.join(base_dir, '..', 'saved_models', 'efficientnet_finetune_best_v2.pth'))
        
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.eval()
            model.to(DEVICE)
            return model
        else:
            st.error(f"Model file not found at: {model_path}")
            return None
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# Tải trước mô hình khi khởi động web
with st.spinner("Đang khởi động hệ thống AI..."):
    model = load_model()


# Giao diện & Điều hướng
st.sidebar.title("Quản lý hệ thống")
menu = st.sidebar.radio("Điều hướng:", ["🔍 Nhận diện Rác", "📊 Lịch sử & Thống kê"])
st.sidebar.markdown("---")
st.sidebar.success("Database: SQLite Connected 🟢")

if menu == "🔍 Nhận diện Rác":
    st.title("♻️ GreenAI - Phân loại rác thải thông minh")
    st.markdown("Tải lên ảnh rác, hệ thống sẽ nhận dạng và đưa ra hướng dẫn xử lý chi tiết.")

    uploaded_file = st.file_uploader("Chọn ảnh (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

    # XỬ LÝ LƯU SESSION
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
        # Nếu là một file ảnh mới hoàn toàn -> Đánh dấu cần đem đi quét AI
        if ('last_uploaded_name' not in st.session_state) or (st.session_state['last_uploaded_name'] != uploaded_file.name):
            st.session_state['uploaded_image'] = image
            st.session_state['last_uploaded_name'] = uploaded_file.name
            st.session_state['need_analyze'] = True

    # Nếu trong Session đã lưu ảnh
    if 'uploaded_image' in st.session_state:
        image = st.session_state['uploaded_image']
        col1, col2 = st.columns([1, 1.2])
        
        with col1:
            st.image(image, caption="Ảnh tải lên", use_container_width=True)
            
        with col2:
            if model:
                # Nếu được đánh dấu là ảnh mới -> Chạy AI và Lưu Database
                if st.session_state.get('need_analyze', False):
                    # Tiền xử lý & Dự đoán
                    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        outputs = model(img_tensor)
                        probs = F.softmax(outputs[0], dim=0)
                        top_probs, top_idxs = torch.topk(probs, 3)
                        
                    # Lưu lại các kết quả vào Session để giữ trạng thái
                    st.session_state['best_class'] = CLASS_NAMES[top_idxs[0].item()]
                    st.session_state['best_conf'] = top_probs[0].item() * 100
                    
                    other_preds = []
                    for i in range(1, 3):
                        idx = top_idxs[i].item()
                        other_preds.append({"name": CLASS_NAMES[idx], "conf": top_probs[i].item() * 100})
                    st.session_state['other_preds'] = other_preds
                    
                    # Ghi log vào Database 1 LẦN DUY NHẤT
                    save_to_db(st.session_state['best_class'], st.session_state['best_conf'])
                    
                    # Đã phân tích xong, tắt cờ để lần sau quay lại không quét lại nữa
                    st.session_state['need_analyze'] = False

                # LẤY THÔNG TIN TỪ SESSION ĐỂ HIỂN THỊ
                best_class = st.session_state['best_class']
                best_conf = st.session_state['best_conf']
                other_preds = st.session_state['other_preds']
                
                # Hiển thị kết quả
                st.subheader("Kết quả nhận diện")
                st.success(f"**Loại rác:** {best_class.upper()}")
                st.metric("Độ tin cậy", f"{best_conf:.2f}%")
                st.progress(int(best_conf))
                
                with st.expander("Các khả năng khác"):
                    for pred in other_preds:
                        st.write(f"- {pred['name']}: {pred['conf']:.2f}%")
                        
                st.markdown("---")
                st.subheader("💡 Hướng dẫn xử lý")
                st.info(ADVICE.get(best_class, "Không có lời khuyên cho loại rác này."))
            else:
                st.error("Không thể tải mô hình. Vui lòng kiểm tra cấu hình.")

elif menu == "📊 Lịch sử & Thống kê":
    st.title("📊 Thống kê Phân loại Rác")
    
    # Lấy dữ liệu từ Database
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM history ORDER BY id DESC", conn)
    conn.close()
    
    if df.empty:
        st.info("Chưa có dữ liệu. Vui lòng quét thử ảnh ở tab 'Nhận diện Rác'.")
    else:
        # Các chỉ số tổng quan
        col1, col2, col3 = st.columns(3)
        col1.metric("Tổng số rác đã quét", f"{len(df)} ảnh")
        col2.metric("Độ tin cậy trung bình", f"{df['confidence'].mean():.2f}%")
        col3.metric("Loại rác quét nhiều nhất", df['trash_type'].mode()[0].upper())
        
        st.markdown("---")
        
        # Vẽ biểu đồ
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Cơ cấu theo Thùng Rác**")
            fig1 = px.pie(df, names='category', hole=0.4)
            fig1.update_traces(hovertemplate='<b>%{label}</b><br>Tỷ lệ: %{percent}<extra></extra>')
            st.plotly_chart(fig1, use_container_width=True)
            
        with c2:
            st.markdown("**Số lượng từng Loại Rác**")
            counts = df['trash_type'].value_counts().reset_index()
            counts.columns = ['trash_type', 'count']
            fig2 = px.bar(counts, x='trash_type', y='count', color='trash_type')
            fig2.update_traces(hovertemplate='<b>%{x}</b><br>Số lượng: %{y} ảnh<extra></extra>')
            st.plotly_chart(fig2, use_container_width=True)
            
        # Bảng nhật ký gần đây
        st.markdown("### 📝 Nhật ký quét gần đây")
        st.dataframe(df[['scan_time', 'trash_type', 'category', 'confidence']].head(10), use_container_width=True)