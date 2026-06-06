import streamlit as st
from PIL import Image
import os
import time
import requests
import base64
import io
from gemini_helper import get_waste_advice
from database import save_to_db, save_feedback

CLASS_NAMES = ['battery', 'cardboard', 'clothes', 'glass', 'metal', 'organic', 'paper', 'plastic', 'shoes', 'trash']

API_URL = "http://127.0.0.1:8000/predict"

def show_scanner():
    st.title("♻️ GreenAI - Phân loại rác thải thông minh")
    st.markdown("Tải lên ảnh rác hoặc chụp trực tiếp từ điện thoại, hệ thống sẽ nhận dạng và đưa ra hướng dẫn xử lý chi tiết.")

    uploaded_file = st.file_uploader("Chọn ảnh hoặc Chụp ảnh mới (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Kiểm duyệt dung lượng file tải lên
        MAX_FILE_SIZE = 5 * 1024 * 1024  # Cấm file > 5 MB
        
        if uploaded_file.size > MAX_FILE_SIZE:
            st.error("⚠️ Tệp tải lên quá lớn! Vui lòng chọn ảnh dưới 5MB để đảm bảo hiệu năng hệ thống.")
        else:
            image = Image.open(uploaded_file).convert('RGB')
            file_id = f"{uploaded_file.name}_{uploaded_file.size}"
            
            if ('last_file_id' not in st.session_state) or (st.session_state['last_file_id'] != file_id):
                st.session_state['uploaded_image'] = image
                st.session_state['uploaded_file_bytes'] = uploaded_file.getvalue()
                st.session_state['last_file_id'] = file_id
                st.session_state['need_analyze'] = True
                st.session_state.pop('heatmap_image', None)
                st.session_state.pop('gemini_result', None)

    if 'uploaded_image' in st.session_state:
        image = st.session_state['uploaded_image']
        col1, col2 = st.columns([1, 1.2])
        
        with col1:
            # Nếu đã phân tích xong và có bản đồ nhiệt
            if not st.session_state.get('need_analyze', True) and st.session_state.get('heatmap_image'):
                opt_orig = "📸 Ảnh gốc"
                opt_heat = "🎯 Vùng nhận diện"
                view_opts = [opt_orig, opt_heat]
                
                # Bộ nhớ ngầm để lưu lựa chọn
                if 'saved_view_index' not in st.session_state:
                    st.session_state['saved_view_index'] = 0

                def update_view_mode():
                    selected_text = st.session_state.get('radio_widget_key', '')
                    if selected_text == "📸 Ảnh gốc":
                        st.session_state['saved_view_index'] = 0
                    elif selected_text == "🎯 Vùng nhận diện":
                        st.session_state['saved_view_index'] = 1
                
                st.radio("Chế độ xem:", 
                         view_opts, 
                         index=st.session_state['saved_view_index'],
                         horizontal=True, 
                         key="radio_widget_key", 
                         on_change=update_view_mode, 
                         label_visibility="collapsed")
                
                if st.session_state['saved_view_index'] == 0:
                    st.image(image, use_container_width=True)
                else:
                    original_size = image.size 
                    heatmap_resized = st.session_state['heatmap_image'].resize(original_size)    
                    st.image(heatmap_resized, use_container_width=True)
                    st.caption("Vùng màu đỏ thể hiện vị trí AI đang tập trung nhìn.")
            else:
                st.image(image, caption="Ảnh đang phân tích", use_container_width=True)
            
        with col2:
            if st.session_state.get('need_analyze', False):
                t0 = time.time()
                try:
                    files = {"file": ("image.jpg", st.session_state['uploaded_file_bytes'], "image/jpeg")}
                    response = requests.post(API_URL, files=files, timeout=30)
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state['time_api'] = (time.time() - t0) * 1000
                        
                        st.session_state['best_class'] = result['best_class']
                        st.session_state['best_conf'] = result['best_conf']
                        st.session_state['other_preds'] = result['other_preds']
                        st.session_state['gradcam_focus'] = result.get('gradcam_focus', 'toàn bộ vật thể')
                        
                        # Giải mã ảnh heatmap từ base64
                        if 'heatmap_base64' in result and result['heatmap_base64']:
                            heatmap_bytes = base64.b64decode(result['heatmap_base64'])
                            st.session_state['heatmap_image'] = Image.open(io.BytesIO(heatmap_bytes))
                        else:
                            st.session_state['heatmap_image'] = None
                        
                        # Lưu ảnh vào máy cục bộ
                        save_dir = 'history_images'
                        os.makedirs(save_dir, exist_ok=True)
                        image_filename = f"{st.session_state['username']}_{int(time.time())}.jpg"
                        image_path = os.path.join(save_dir, image_filename)
                        image.save(image_path)
                        
                        st.session_state['current_image_path'] = image_path
                        
                        # Lưu cơ sở dữ liệu
                        t2 = time.time()
                        save_to_db(st.session_state['username'], st.session_state['best_class'], st.session_state['best_conf'], image_path)
                        st.session_state['time_db'] = (time.time() - t2) * 1000
                        
                        st.session_state['need_analyze'] = False
                        st.session_state['feedback_submitted'] = False
                        st.session_state['corrected_label'] = None
                        st.session_state['show_correction'] = False
                        st.rerun()
                    else:
                        st.error("❌ Lỗi từ máy chủ AI. Vui lòng thử lại sau.")
                        st.session_state['need_analyze'] = False
                        
                except requests.exceptions.Timeout:
                    st.error("⏱️ Máy chủ AI phản hồi quá chậm (>30 giây). Vui lòng thử lại.")
                    st.session_state['need_analyze'] = False
                except Exception as e:
                    st.error(f"❌ Không thể kết nối đến Backend API. Hãy chắc chắn bạn đã chạy api.py. Lỗi: {e}")
                    st.session_state['need_analyze'] = False

            # Hiển thị kết quả nếu đã phân tích xong
            if not st.session_state.get('need_analyze', True) and 'best_class' in st.session_state:
                best_class = st.session_state['best_class']
                best_conf = st.session_state['best_conf']
                other_preds = st.session_state['other_preds']
                
                st.subheader("Kết quả nhận diện")
                st.success(f"**Loại rác:** {best_class.upper()}")
                
                st.metric("Độ tin cậy", f"{best_conf:.2f}%")
                st.progress(int(best_conf))
                
                with st.expander("Các khả năng khác"):
                    for pred in other_preds:
                        st.write(f"- {pred['name']}: {pred['conf']:.2f}%")

                st.markdown("---")
                st.subheader("💡 Chuyên gia AI tư vấn")
                
                # Chỉ gọi API nếu chưa có kết quả
                if 'gemini_result' not in st.session_state:
                    with st.spinner("Gemini đang tư vấn..."):
                        gradcam_focus = st.session_state.get('gradcam_focus', 'toàn bộ vật thể')
                        confidence_score = st.session_state.get('best_conf', 0.0)
                        
                        fresh_result = get_waste_advice(
                            image=image, 
                            waste_class=best_class, 
                            confidence=confidence_score, 
                            gradcam_focus=gradcam_focus
                        )
                        st.session_state['gemini_result'] = fresh_result

                # Lấy dữ liệu từ kết sắt
                saved_data = st.session_state['gemini_result']
                
                st.markdown("**🔍 Mô tả từ AI:**")
                st.write(saved_data.get("description", ""))
                
                st.markdown("**💡 Hướng dẫn xử lý:**")
                st.success(saved_data.get("advice", ""))

                # Dev mode: Bộ đo lường hiệu năng
                with st.expander("⚙️ Dành cho Nhà phát triển (Hiệu năng API)"):
                    t_api = st.session_state.get('time_api', 0)
                    t_db = st.session_state.get('time_db', 0)
                    
                    st.caption(f"🌐 Độ trễ mạng & AI Suy luận (API Latency): **{t_api:.0f}ms**")
                    st.caption(f"💾 Thao tác I/O & CSDL: **{t_db:.0f}ms**")

                # Giao diện phản hồi
                st.markdown("---")
                if st.session_state.get('feedback_submitted', False):
                    if st.session_state.get('corrected_label'):
                        st.info(f"✅ Bạn đã đính chính rác này là: **{st.session_state['corrected_label'].upper()}**")
                    else:
                        st.success("✅ Bạn đã xác nhận AI dự đoán đúng!")
                else:
                    st.write("**AI dự đoán có chính xác không?**")
                    f_col1, f_col2 = st.columns(2)
                    
                    with f_col1:
                        if st.button("👍 Đúng quá!", use_container_width=True):
                            save_feedback(st.session_state['username'], best_class, best_class, True, st.session_state.get('current_image_path'))
                            st.session_state['feedback_submitted'] = True
                            st.session_state['corrected_label'] = None
                            st.rerun()
                            
                    with f_col2:
                        if st.button("👎 Sai rồi!", use_container_width=True):
                            st.session_state['show_correction'] = True
                            
                    if st.session_state.get('show_correction', False):
                        st.warning("Rất tiếc vì hệ thống nhầm lẫn. Xin hãy giúp chúng tôi cải thiện!")
                        true_label = st.selectbox("Theo bạn, đây thực chất là rác gì?", CLASS_NAMES)
                        
                        if st.button("Gửi đáp án đúng"):
                            save_feedback(st.session_state['username'], best_class, true_label, False, st.session_state.get('current_image_path'))
                            st.session_state['feedback_submitted'] = True
                            st.session_state['corrected_label'] = true_label
                            st.session_state['show_correction'] = False
                            st.rerun()