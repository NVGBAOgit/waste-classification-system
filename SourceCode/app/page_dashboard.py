import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import os
from PIL import Image
from database import DB_NAME

def show_dashboard():
    st.title("📊 Thống kê Phân loại rác")
    current_user = st.session_state.get('username', 'Guest')
    
    try:
        conn = sqlite3.connect(DB_NAME)
        # Lấy dữ liệu lịch sử phân loại
        df = pd.read_sql_query("SELECT * FROM history WHERE username = ? ORDER BY id DESC", conn, params=(current_user,))
        
        # Lấy dữ liệu phản hồi
        try:
            df_feedback = pd.read_sql_query("SELECT * FROM feedback WHERE username = ? ORDER BY id DESC", conn, params=(current_user,))
        except:
            df_feedback = pd.DataFrame()
            
        conn.close()

        if df.empty:
            st.info("Chưa có dữ liệu. Hãy tải ảnh lên ở mục 'Nhận diện Rác' để bắt đầu phân tích!")
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("Tổng số ảnh đã quét", f"{len(df)} ảnh")
            col2.metric("Độ tin cậy trung bình", f"{df['confidence'].mean():.2f}%")
            col3.metric("Loại rác quét nhiều nhất", df['trash_type'].mode()[0].upper())
            
            st.markdown("---")
            
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
                
            st.markdown("### 📝 Lịch sử phân loại rác")
            st.caption("💡 Mẹo: Nhấn chọn một dòng bất kỳ trong bảng dưới đây để xem lại ảnh và chi tiết!")
            
            # Chỉ hiển thị các cột cần thiết cho bảng
            display_history = df[['scan_time', 'trash_type', 'category', 'confidence']]
            
            # Khởi tạo bảng tương tác với chiều cao cố định
            event = st.dataframe(
                display_history, 
                use_container_width=True, 
                height=300,
                on_select="rerun", 
                selection_mode="single-row"
            )
            
            # Xử lý sự kiện nếu người dùng Click vào dòng
            selected_rows = event.selection.rows
            if selected_rows:
                index = selected_rows[0]
                selected_data = df.iloc[index] # Lấy dữ liệu từ bảng history
                
                st.markdown("#### 🔍 Chi tiết bản ghi được chọn")
                col_img, col_txt = st.columns([1, 1.5])
                
                with col_img:
                    # Truy xuất đường dẫn ảnh đã lưu
                    path = selected_data.get('image_path')
                    if pd.notna(path) and os.path.exists(path):
                        st.image(Image.open(path), caption="Ảnh thực tế", use_container_width=True)
                    else:
                        st.warning("Ảnh này không tồn tại trên hệ thống (Bản ghi cũ).")
                
                with col_txt:
                    st.write(f"📅 **Thời gian:** {selected_data['scan_time']}")
                    st.write(f"🤖 **Hệ thống AI dự đoán:** {selected_data['trash_type'].upper()}")
                    st.write(f"🗑️ **Phân loại:** {selected_data['category']}")
                    st.success(f"Độ tự tin: {selected_data['confidence']:.2f}%")

        st.markdown("---")
        st.subheader("📝 Lịch sử phản hồi của bạn")

        if not df_feedback.empty:
            display_fb = df_feedback[['timestamp', 'predicted_class', 'true_class', 'is_correct']]
            st.dataframe(display_fb, use_container_width=True, height=250)
        else:
            st.info("Chưa có phản hồi nào được ghi nhận.")

    except Exception as e:
        st.error(f"Đã xảy ra lỗi khi tải dữ liệu: {e}")