import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import sys
import os
import time
from database import save_to_db, save_feedback

CLASS_NAMES = ['battery', 'cardboard', 'clothes', 'glass', 'metal', 'organic', 'paper', 'plastic', 'shoes', 'trash']

# TỪ ĐIỂN SONG NGỮ CHO HƯỚNG DẪN XỬ LÝ
ADVICE_VI = {
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

ADVICE_EN = {
    'battery': "**BATTERIES – HAZARDOUS WASTE**\n\n📌 **Disposal:**\n- Do not mix with regular trash.\n- Tape both ends.\n- Take to the nearest battery collection point.\n\n**Bin:** Special bin (hazardous).",
    'cardboard': "**CARDBOARD – RECYCLABLE**\n\n📌 **Disposal:**\n- Remove tape and staples.\n- Flatten to save space.\n- Keep dry and free of grease.\n\n**Bin:** Blue (recyclable).",
    'clothes': "**OLD CLOTHES – NON-RECYCLABLE**\n\n📌 **Disposal:**\n- If in good condition: donate or sell.\n- If heavily damaged: cut into pieces, put in a sealed bag.\n- Do not mix with organic waste.\n\n**Bin:** Orange (non-recyclable).",
    'glass': "**GLASS – RECYCLABLE**\n\n📌 **Disposal:**\n- Empty liquids.\n- Rinse and dry.\n- Wrap in newspaper if broken to prevent injury.\n\n**Bin:** Blue (recyclable).",
    'metal': "**METAL – RECYCLABLE**\n\n📌 **Disposal:**\n- Empty food/liquids.\n- Rinse and crush cans.\n- Remove caps (if plastic).\n\n**Bin:** Blue (recyclable).",
    'organic': "**ORGANIC WASTE**\n\n📌 **Disposal:**\n- Put directly in the bin, no plastic bags.\n- Compost at home if possible.\n- Do not mix with plastic or glass.\n\n**Bin:** Brown (organic).",
    'paper': "**PAPER – RECYCLABLE**\n\n📌 **Disposal:**\n- Clean paper, free of food or grease.\n- Fold or shred.\n- Soiled paper goes to non-recyclable.\n\n**Bin:** Blue (recyclable).",
    'plastic': "**PLASTIC – RECYCLABLE**\n\n📌 **Disposal:**\n- Empty liquids, rinse.\n- Crush to save space.\n- Remove caps (caps are also recyclable).\n\n**Bin:** Blue (recyclable).",
    'shoes': "**OLD SHOES – NON-RECYCLABLE**\n\n📌 **Disposal:**\n- Tie pairs together.\n- If usable: donate.\n- If damaged: put in non-recyclable bin.\n\n**Bin:** Orange (non-recyclable).",
    'trash': "**MIXED WASTE – NON-RECYCLABLE**\n\n📌 **Disposal:**\n- Tie garbage bags securely.\n- Do not mix with metal, glass, batteries, or organic waste.\n- Destined for landfill or incineration.\n\n**Bin:** Orange (non-recyclable)."
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        return None
    except Exception as e:
        return None

def show_scanner():
    lang = st.session_state.get('lang', 'vi')

    with st.spinner("Đang khởi động hệ thống AI..." if lang=='vi' else "Initializing AI system..."):
        model = load_model()

    st.title("♻️ GreenAI - Phân loại rác thải thông minh" if lang=='vi' else "♻️ GreenAI - Smart Waste Classification")
    st.markdown("Tải lên ảnh rác hoặc chụp trực tiếp từ điện thoại, hệ thống sẽ nhận dạng và đưa ra hướng dẫn xử lý chi tiết." if lang=='vi' else "Upload an image of waste or take a photo directly from your phone, the system will identify it and provide detailed disposal instructions.")

    upl_text = "Chọn ảnh hoặc Chụp ảnh mới (jpg, jpeg, png)" if lang=='vi' else "Choose or take a new photo (jpg, jpeg, png)"
    uploaded_file = st.file_uploader(upl_text, type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        
        if ('last_file_id' not in st.session_state) or (st.session_state['last_file_id'] != file_id):
            st.session_state['uploaded_image'] = image
            st.session_state['last_file_id'] = file_id
            st.session_state['need_analyze'] = True

    if 'uploaded_image' in st.session_state:
        image = st.session_state['uploaded_image']
        col1, col2 = st.columns([1, 1.2])
        
        with col1:
            st.image(image, caption="Ảnh đang phân tích" if lang=='vi' else "Image being analyzed", use_container_width=True)
            
        with col2:
            if model:
                if st.session_state.get('need_analyze', False):
                    # THỜI GIAN TIỀN XỬ LÝ
                    t0 = time.time()
                    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
                    st.session_state['time_prep'] = (time.time() - t0) * 1000
                    
                    # THỜI GIAN AI SUY LUẬN
                    t1 = time.time()
                    with torch.no_grad():
                        outputs = model(img_tensor)
                        probs = F.softmax(outputs[0], dim=0)
                        top_probs, top_idxs = torch.topk(probs, 3)
                    st.session_state['time_inf'] = (time.time() - t1) * 1000
                        
                    st.session_state['best_class'] = CLASS_NAMES[top_idxs[0].item()]
                    st.session_state['best_conf'] = top_probs[0].item() * 100
                    
                    other_preds = []
                    for i in range(1, 3):
                        idx = top_idxs[i].item()
                        other_preds.append({"name": CLASS_NAMES[idx], "conf": top_probs[i].item() * 100})
                    st.session_state['other_preds'] = other_preds
                    
                    # LƯU ẢNH VÀO MÁY CỤC BỘ
                    save_dir = 'history_images'
                    os.makedirs(save_dir, exist_ok=True)
                    image_filename = f"{st.session_state['username']}_{int(time.time())}.jpg"
                    image_path = os.path.join(save_dir, image_filename)
                    image.save(image_path)
                    
                    st.session_state['current_image_path'] = image_path
                    
                    # THỜI GIAN GHI CƠ SỞ DỮ LIỆU
                    t2 = time.time()
                    save_to_db(st.session_state['username'], st.session_state['best_class'], st.session_state['best_conf'], image_path)
                    st.session_state['time_db'] = (time.time() - t2) * 1000
                    
                    st.session_state['need_analyze'] = False
                    st.session_state['feedback_submitted'] = False
                    st.session_state['corrected_label'] = None
                    st.session_state['show_correction'] = False

                best_class = st.session_state['best_class']
                best_conf = st.session_state['best_conf']
                other_preds = st.session_state['other_preds']
                
                st.subheader("Kết quả nhận diện" if lang=='vi' else "Detection Results")
                
                type_lbl = "**Loại rác:**" if lang=='vi' else "**Waste Type:**"
                st.success(f"{type_lbl} {best_class.upper()}")
                
                conf_lbl = "Độ tin cậy" if lang=='vi' else "Confidence"
                st.metric(conf_lbl, f"{best_conf:.2f}%")
                st.progress(int(best_conf))
                
                with st.expander("Các khả năng khác" if lang=='vi' else "Other Possibilities"):
                    for pred in other_preds:
                        st.write(f"- {pred['name']}: {pred['conf']:.2f}%")
                        
                st.markdown("---")
                st.subheader("💡 Hướng dẫn xử lý" if lang=='vi' else "💡 Disposal Guide")
                if lang == 'vi':
                    st.info(ADVICE_VI.get(best_class, "Không có lời khuyên cho loại rác này."))
                else:
                    st.info(ADVICE_EN.get(best_class, "No advice available for this waste type."))

                # ⚙️ DEV MODE: BỘ ĐO LƯỜNG HIỆU NĂNG
                dev_title = "⚙️ Dành cho Nhà phát triển (Hiệu năng AI)" if lang=='vi' else "⚙️ Developer Mode (AI Profiling)"
                with st.expander(dev_title):
                    t_prep = st.session_state.get('time_prep', 0)
                    t_inf = st.session_state.get('time_inf', 0)
                    t_db = st.session_state.get('time_db', 0)
                    
                    if lang == 'vi':
                        st.caption(f"⏱️ Tiền xử lý Tensor: **{t_prep:.0f}ms**")
                        st.caption(f"🧠 Mô hình AI Suy luận: **{t_inf:.0f}ms**")
                        st.caption(f"💾 Thao tác I/O & CSDL: **{t_db:.0f}ms**")
                    else:
                        st.caption(f"⏱️ Tensor Preprocessing: **{t_prep:.0f}ms**")
                        st.caption(f"🧠 AI Model Inference: **{t_inf:.0f}ms**")
                        st.caption(f"💾 I/O & Database Ops: **{t_db:.0f}ms**")

                # GIAO DIỆN PHẢN HỒI 
                st.markdown("---")
                if st.session_state.get('feedback_submitted', False):
                    if st.session_state.get('corrected_label'):
                        if lang == 'vi':
                            st.info(f"✅ Bạn đã đính chính rác này là: **{st.session_state['corrected_label'].upper()}**")
                        else:
                            st.info(f"✅ You corrected this waste as: **{st.session_state['corrected_label'].upper()}**")
                    else:
                        st.success("✅ Bạn đã xác nhận AI dự đoán đúng!" if lang=='vi' else "✅ You confirmed the AI prediction is correct!")
                else:
                    st.write("**AI dự đoán có chính xác không?**" if lang=='vi' else "**Is the AI prediction correct?**")
                    f_col1, f_col2 = st.columns(2)
                    
                    with f_col1:
                        if st.button("👍 Đúng quá!" if lang=='vi' else "👍 Spot on!", use_container_width=True):
                            save_feedback(st.session_state['username'], best_class, best_class, True, st.session_state.get('current_image_path'))
                            st.session_state['feedback_submitted'] = True
                            st.session_state['corrected_label'] = None
                            st.rerun() 
                            
                    with f_col2:
                        if st.button("👎 Sai rồi!" if lang=='vi' else "👎 Incorrect!", use_container_width=True):
                            st.session_state['show_correction'] = True
                            
                    if st.session_state.get('show_correction', False):
                        st.warning("Rất tiếc vì hệ thống nhầm lẫn. Xin hãy giúp chúng tôi cải thiện!" if lang=='vi' else "Sorry for the mistake. Please help us improve!")
                        sel_lbl = "Theo bạn, đây thực chất là rác gì?" if lang=='vi' else "What is the actual waste type?"
                        true_label = st.selectbox(sel_lbl, CLASS_NAMES)
                        
                        sub_lbl = "Gửi đáp án đúng" if lang=='vi' else "Submit correct answer"
                        if st.button(sub_lbl):
                            save_feedback(st.session_state['username'], best_class, true_label, False, st.session_state.get('current_image_path'))
                            st.session_state['feedback_submitted'] = True
                            st.session_state['corrected_label'] = true_label
                            st.session_state['show_correction'] = False
                            st.rerun() 
            else:
                st.error("Không thể tải mô hình. Vui lòng kiểm tra cấu hình." if lang=='vi' else "Failed to load model. Please check configuration.")