import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import sys
import os

CLASS_NAMES = [
    'battery', 'cardboard', 'clothes', 'glass', 'metal', 'organic', 'paper', 'plastic', 'shoes', 'trash'
]

# Lời khuyên xử lý cho từng loại rác
ADVICE = {
    'battery': "**PIN – RÁC NGUY HẠI**\n\n"
               "📌 **Cách xử lý:**\n"
               "- Không vứt chung rác thông thường.\n"
               "- Bọc kín hai đầu cực bằng băng keo.\n"
               "- Mang đến điểm thu gom pin, ắc quy gần nhất.\n"
               "\n**Thùng rác:** Thùng đặc biệt (rác nguy hại).",

    'cardboard': "**BÌA CARTON – RÁC TÁI CHẾ**\n\n"
                 "📌 **Cách xử lý:**\n"
                 "- Tháo băng keo, gỡ kim ghim.\n"
                 "- Bóp dẹp hoặc xếp chồng để tiết kiệm diện tích.\n"
                 "- Giữ khô ráo, không bị ướt hoặc dính dầu mỡ.\n"
                 "\n**Thùng rác:** Màu xanh (tái chế).",

    'clothes': "**QUẦN ÁO CŨ – RÁC VÔ CƠ**\n\n"
               "📌 **Cách xử lý:**\n"
               "- Nếu còn tốt: ủng hộ hoặc bán đồ second-hand.\n"
               "- Nếu hỏng nặng: cắt nhỏ, bỏ vào túi kín.\n"
               "- Không trộn với rác hữu cơ.\n"
               "\n**Thùng rác:** Màu cam (vô cơ).",

    'glass': "**THỦY TINH – RÁC TÁI CHẾ**\n\n"
             "📌 **Cách xử lý:**\n"
             "- Đổ hết chất lỏng bên trong.\n"
             "- Rửa sạch và để ráo nước.\n"
             "- Dùng giấy báo bọc ngoài nếu chai vỡ để tránh nguy hiểm.\n"
             "\n**Thùng rác:** Màu xanh (tái chế).",

    'metal': "**KIM LOẠI – RÁC TÁI CHẾ**\n\n"
             "📌 **Cách xử lý:**\n"
             "- Đổ hết thức ăn/nước bên trong.\n"
             "- Rửa sạch và bóp dẹp lon/hộp.\n"
             "- Tháo nắp riêng (nếu là nắp nhựa).\n"
             "\n**Thùng rác:** Màu xanh (tái chế).",

    'organic': "**RÁC HỮU CƠ**\n\n"
               "📌 **Cách xử lý:**\n"
               "- Bỏ trực tiếp vào thùng, không cần túi nilon.\n"
               "- Có thể ủ phân compost tại nhà nếu có điều kiện.\n"
               "- Không lẫn tạp chất như nhựa, thủy tinh.\n"
               "\n**Thùng rác:** Màu nâu (hữu cơ).",

    'paper': "**GIẤY – RÁC TÁI CHẾ**\n\n"
             "📌 **Cách xử lý:**\n"
             "- Giấy sạch, không dính dầu mỡ hoặc thức ăn.\n"
             "- Gấp gọn hoặc cắt nhỏ.\n"
             "- Giấy bẩn (dùng lau, thấm dầu) bỏ vào rác vô cơ.\n"
             "\n**Thùng rác:** Màu xanh (tái chế).",

    'plastic': "**NHỰA – RÁC TÁI CHẾ**\n\n"
               "📌 **Cách xử lý:**\n"
               "- Đổ cạn nước, rửa sạch.\n"
               "- Bóp dẹp chai/hộp để tiết kiệm không gian.\n"
               "- Bỏ nắp riêng (nắp nhựa cũng tái chế được).\n"
               "\n**Thùng rác:** Màu xanh (tái chế).",

    'shoes': "**GIÀY DÉP CŨ – RÁC VÔ CƠ**\n\n"
             "📌 **Cách xử lý:**\n"
             "- Buộc cặp đôi lại với nhau.\n"
             "- Nếu còn dùng được: tặng cho người có nhu cầu.\n"
             "- Nếu hỏng: bỏ vào thùng rác vô cơ.\n"
             "\n**Thùng rác:** Màu cam (vô cơ).",

    'trash': "**RÁC HỖN HỢP – VÔ CƠ**\n\n"
             "📌 **Cách xử lý:**\n"
             "- Buộc kín miệng túi rác.\n"
             "- Không lẫn kim loại, thủy tinh, pin hay rác hữu cơ.\n"
             "- Đây là rác sẽ được chôn lấp hoặc đốt.\n"
             "\n**Thùng rác:** Màu cam (vô cơ)."
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tiền xử lý ảnh
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@st.cache_resource
def load_model():
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
            st.error(f"Không tìm thấy model tại: {model_path}")
            return None
    except Exception as e:
        st.error(f"Lỗi nạp model: {e}")
        return None

st.set_page_config(page_title="GreenAI - Phân loại rác", page_icon="♻️", layout="wide")

# TẢI NGẦM MODEL NGAY KHI VÀO TRANG 
with st.spinner("Đang khởi động hệ thống AI..."):
    model = load_model()

st.title("♻️ GreenAI - Phân loại rác thải thông minh")
st.markdown("Tải lên ảnh rác, hệ thống sẽ nhận dạng và đưa ra hướng dẫn xử lý chi tiết.")

uploaded_file = st.file_uploader("Chọn ảnh (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.image(image, caption="Ảnh của bạn", use_container_width=True)
        
    with col2:
        if model:
            # Tiền xử lý
            img_tensor = transform(image).unsqueeze(0).to(DEVICE)
            
            # Dự đoán
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = F.softmax(outputs[0], dim=0)
                top_probs, top_idxs = torch.topk(probs, 3)
                
            # Kết quả cao nhất
            best_idx = top_idxs[0].item()
            best_class = CLASS_NAMES[best_idx]
            best_conf = top_probs[0].item() * 100
            
            # Hiển thị kết quả
            st.subheader("Kết quả nhận diện")
            st.success(f"**Loại rác:** {best_class.upper()}")
            st.metric("Độ tin cậy", f"{best_conf:.2f}%")
            st.progress(int(best_conf))
            
            # Hiển thị top 2, 3
            with st.expander("Các khả năng khác"):
                for i in range(1, 3):
                    idx = top_idxs[i].item()
                    name = CLASS_NAMES[idx]
                    conf = top_probs[i].item() * 100
                    st.write(f"- {name}: {conf:.2f}%")
                    
            # Lời khuyên
            st.markdown("---")
            st.subheader("💡 Hướng dẫn xử lý")
            st.info(ADVICE.get(best_class, "Không có lời khuyên cho loại rác này."))
        else:
            st.error("Không thể tải mô hình. Vui lòng kiểm tra file .pth.")
