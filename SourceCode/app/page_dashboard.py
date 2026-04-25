import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
from database import DB_NAME

def show_dashboard():
    st.title("📊 Thống kê Phân loại rác")
    
    conn = sqlite3.connect(DB_NAME)
    current_user = st.session_state['username']
    df = pd.read_sql_query("SELECT * FROM history WHERE username = ? ORDER BY id DESC", conn, params=(current_user,))
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
        st.dataframe(df[['scan_time', 'trash_type', 'category', 'confidence']].head(10), use_container_width=True)