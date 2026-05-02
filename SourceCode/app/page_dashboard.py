import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
from database import DB_NAME # Sử dụng DB_NAME thống nhất từ file database.py

def show_dashboard():
    st.title("📊 Thống kê Phân loại rác")
    
    current_user = st.session_state.get('username', 'Guest')
    
    try:
        # Mở kết nối đến cơ sở dữ liệu
        conn = sqlite3.connect(DB_NAME)
        
        # Lấy dữ liệu lịch sử phân loại
        df = pd.read_sql_query("SELECT * FROM history WHERE username = ? ORDER BY id DESC", conn, params=(current_user,))
        
        # Lấy dữ liệu phản hồi (Feedback) của riêng người dùng này
        # Sử dụng khối try-except nhỏ cho bảng feedback để tránh lỗi nếu bảng chưa tồn tại
        try:
            df_feedback = pd.read_sql_query("SELECT * FROM feedback WHERE username = ? ORDER BY id DESC", conn, params=(current_user,))
        except:
            df_feedback = pd.DataFrame() # Nếu lỗi (bảng chưa có), coi như chưa có dữ liệu
            
        conn.close()

        if df.empty:
            st.info("Chưa có dữ liệu. Hãy tải ảnh lên ở mục 'Nhận diện Rác' để bắt đầu phân tích!")
        else:
            # HIỂN THỊ CÁC CHỈ SỐ TỔNG QUAN
            col1, col2, col3 = st.columns(3)
            col1.metric("Tổng số ảnh đã quét", f"{len(df)} ảnh")
            col2.metric("Độ tin cậy trung bình", f"{df['confidence'].mean():.2f}%")
            col3.metric("Loại rác quét nhiều nhất", df['trash_type'].mode()[0].upper())
            
            st.markdown("---")
            
            # HIỂN THỊ BIỂU ĐỒ
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
            st.dataframe(df[['scan_time', 'trash_type', 'category', 'confidence']].head(10), use_container_width=True)

        st.markdown("---")
        st.subheader("📝 Lịch sử phản hồi của bạn")
        
        if not df_feedback.empty:
            # Chỉ hiển thị các cột quan trọng để giao diện sạch sẽ
            display_fb = df_feedback[['timestamp', 'predicted_class', 'true_class', 'is_correct']]
            st.dataframe(display_fb, use_container_width=True)
        else:
            st.info("Chưa có phản hồi nào từ bạn được ghi nhận.")

    except Exception as e:
        st.error(f"Đã xảy ra lỗi khi tải dữ liệu: {e}")