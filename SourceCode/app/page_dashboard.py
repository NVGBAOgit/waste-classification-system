import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import os
from PIL import Image
from database import DB_NAME, delete_history_and_feedback, delete_all_user_data

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
            
            # Khởi tạo ngày nhỏ nhất và lớn nhất trong dữ liệu
            df['scan_date'] = pd.to_datetime(df['scan_time'], errors='coerce').dt.date
            min_date = df['scan_date'].min()
            max_date = df['scan_date'].max()

            # Chia làm 3 cột (tỷ lệ 2 : 1.5 : 1.5) để giao diện thoáng mắt
            filter_col1, filter_col2, filter_col3 = st.columns([2, 1.5, 1.5])
            
            with filter_col1:
                filter_options = ["Tất cả", "Tái Chế", "Hữu Cơ", "Nguy Hại", "Vô Cơ"]
                selected_category = st.selectbox("📁 Lọc theo Thùng rác:", filter_options)
                
            with filter_col2:
                start_date = st.date_input("📅 Từ ngày:", value=min_date, min_value=min_date, max_value=max_date)
                
            with filter_col3:
                end_date = st.date_input("📅 Đến ngày:", value=max_date, min_value=min_date, max_value=max_date)

            # Cảnh báo trải nghiệm người dùng (UX Validation)
            if start_date > end_date:
                st.warning("⚠️ 'Từ ngày' không thể lớn hơn 'Đến ngày'. Vui lòng chọn lại!")
            
            # Áp dụng bộ lọc vào DataFrame
            filtered_df = df.copy()
            
            # Lọc theo nhóm thùng rác
            if selected_category != "Tất cả":
                filtered_df = filtered_df[filtered_df['category'] == selected_category]
                
            # Lọc theo khoảng thời gian
            if start_date <= end_date:
                filtered_df = filtered_df[(filtered_df['scan_date'] >= start_date) & (filtered_df['scan_date'] <= end_date)]
                
            st.caption("💡 Mẹo: Nhấn chọn một dòng bất kỳ trong bảng dưới đây để xem lại ảnh, chi tiết và tùy chọn XÓA!")
            
            display_history = filtered_df[['scan_time', 'trash_type', 'category', 'confidence']]
            
            event = st.dataframe(
                display_history, 
                use_container_width=True, 
                height=300, 
                on_select="rerun", 
                selection_mode="single-row",
                hide_index=True 
            )
            
            # Xử lý click chọn dòng
            selected_rows = event.selection.rows
            if selected_rows:
                index = selected_rows[0]
                selected_data = filtered_df.iloc[index] 
                
                st.markdown("#### 🔍 Chi tiết bản ghi được chọn")
                col_img, col_txt = st.columns([1, 1.5])
                
                path = selected_data.get('image_path')
                
                with col_img:
                    if pd.notna(path) and os.path.exists(path):
                        st.image(Image.open(path), caption="Ảnh thực tế", use_container_width=True)
                    else:
                        st.warning("Ảnh này không tồn tại trên hệ thống.")
                
                with col_txt:
                    st.write(f"📅 **Thời gian:** {selected_data['scan_time']}")
                    st.write(f"🤖 **Hệ thống AI dự đoán:** {selected_data['trash_type'].upper()}")
                    st.write(f"🗑️ **Phân loại:** {selected_data['category']}")
                    st.success(f"Độ tự tin: {selected_data['confidence']:.2f}%")
                    
                    if st.button("🗑️ Xóa bản ghi này", type="primary"):
                        try:
                            if pd.notna(path) and os.path.exists(path):
                                os.remove(path)
                            record_id = int(selected_data['id'])
                            delete_history_and_feedback(record_id, path)
                            st.success("Đã xóa bản ghi thành công!")
                            st.rerun() 
                        except Exception as e:
                            st.error(f"Lỗi khi xóa: {e}")

        st.markdown("---")
        st.subheader("📝 Lịch sử phản hồi của bạn")

        if not df_feedback.empty:
            display_fb = df_feedback[['timestamp', 'predicted_class', 'true_class', 'is_correct']]
            st.dataframe(display_fb, use_container_width=True, height=250, hide_index=True)
        else:
            st.info("Chưa có phản hồi nào được ghi nhận.")
            
        if not df.empty:
            st.markdown("---")
            st.subheader("🔴 Xóa dữ liệu")
            
            if st.button("XÓA TOÀN BỘ DỮ LIỆU CỦA TÔI", type="primary", use_container_width=True):
                try:
                    # Xóa file ảnh vật lý
                    for path in df['image_path'].dropna():
                        if path and os.path.exists(path):
                            os.remove(path)
                    
                    # Xóa Database
                    delete_all_user_data(current_user)
                    
                    st.success("Đã dọn dẹp sạch sẽ toàn bộ dữ liệu! Đang tải lại...")
                    st.rerun()
                except Exception as e:
                    st.error(f"Lỗi khi xóa dữ liệu: {e}")
                    
            st.warning("⚠️ Cảnh báo: Thao tác này sẽ xóa vĩnh viễn toàn bộ lịch sử quét ảnh và lịch sử phản hồi của bạn. Không thể hoàn tác!")

    except Exception as e:
        st.error(f"Đã xảy ra lỗi khi tải dữ liệu: {e}")