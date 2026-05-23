import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import os
from PIL import Image
from database import DB_NAME, delete_history_and_feedback, delete_all_user_data

def show_dashboard():
    lang = st.session_state.get('lang', 'vi')

    st.title("📊 Thống kê Phân loại rác" if lang=='vi' else "📊 Waste Classification Statistics")
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
            empty_msg = "Chưa có dữ liệu. Hãy tải ảnh lên ở mục 'Nhận diện Rác' để bắt đầu phân tích!" if lang=='vi' else "No data yet. Upload an image in 'Scanner' to begin!"
            st.info(empty_msg)
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("Tổng số ảnh đã quét" if lang=='vi' else "Total Scanned", f"{len(df)} ảnh" if lang=='vi' else f"{len(df)} images")
            col2.metric("Độ tin cậy trung bình" if lang=='vi' else "Avg Confidence", f"{df['confidence'].mean():.2f}%")
            col3.metric("Loại rác quét nhiều nhất" if lang=='vi' else "Most Frequent", df['trash_type'].mode()[0].upper())
            
            st.markdown("---")
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Cơ cấu theo Thùng Rác**" if lang=='vi' else "**Breakdown by Bin**")
                df_pie = df.copy()
                if lang == 'en':
                    cat_map = {"Tái Chế": "Recyclable", "Hữu Cơ": "Organic", "Nguy Hại": "Hazardous", "Vô Cơ": "Non-recyclable"}
                    df_pie['category'] = df_pie['category'].map(cat_map).fillna(df_pie['category'])
                
                fig1 = px.pie(df_pie, names='category', hole=0.4)
                fig1.update_traces(hovertemplate='<b>%{label}</b><br>Tỷ lệ: %{percent}<extra></extra>' if lang=='vi' else '<b>%{label}</b><br>Ratio: %{percent}<extra></extra>')
                st.plotly_chart(fig1, use_container_width=True)
                
            with c2:
                st.markdown("**Số lượng từng Loại Rác**" if lang=='vi' else "**Count by Waste Type**")
                counts = df['trash_type'].value_counts().reset_index()
                counts.columns = ['trash_type', 'count']
                fig2 = px.bar(counts, x='trash_type', y='count', color='trash_type')
                fig2.update_traces(hovertemplate='<b>%{x}</b><br>Số lượng: %{y}<extra></extra>' if lang=='vi' else '<b>%{x}</b><br>Count: %{y}<extra></extra>')
                st.plotly_chart(fig2, use_container_width=True)
                
            st.markdown("### 📝 Lịch sử phân loại rác" if lang=='vi' else "### 📝 Classification History")
            
            df['scan_date'] = pd.to_datetime(df['scan_time'], errors='coerce').dt.date
            min_date = df['scan_date'].min()
            max_date = df['scan_date'].max()

            filter_col1, filter_col2, filter_col3 = st.columns([2, 1.5, 1.5])
            
            with filter_col1:
                filter_opts_vi = ["Tất cả", "Tái Chế", "Hữu Cơ", "Nguy Hại", "Vô Cơ"]
                filter_opts_en = ["All", "Recyclable", "Organic", "Hazardous", "Non-recyclable"]
                opts = filter_opts_vi if lang == 'vi' else filter_opts_en
                sel_lbl = "📁 Lọc theo Thùng rác:" if lang=='vi' else "📁 Filter by Bin:"
                selected_category_ui = st.selectbox(sel_lbl, opts)
                
                if lang == 'en':
                    idx = filter_opts_en.index(selected_category_ui)
                    selected_category = filter_opts_vi[idx]
                else:
                    selected_category = selected_category_ui
                
            with filter_col2:
                start_date = st.date_input("📅 Từ ngày:" if lang=='vi' else "📅 From:", value=min_date, min_value=min_date, max_value=max_date)
                
            with filter_col3:
                end_date = st.date_input("📅 Đến ngày:" if lang=='vi' else "📅 To:", value=max_date, min_value=min_date, max_value=max_date)

            if start_date > end_date:
                st.warning("⚠️ 'Từ ngày' không thể lớn hơn 'Đến ngày'. Vui lòng chọn lại!" if lang=='vi' else "⚠️ 'From' date cannot be after 'To' date. Please reselect!")
            
            filtered_df = df.copy()
            
            if selected_category != "Tất cả":
                filtered_df = filtered_df[filtered_df['category'] == selected_category]
                
            if start_date <= end_date:
                filtered_df = filtered_df[(filtered_df['scan_date'] >= start_date) & (filtered_df['scan_date'] <= end_date)]
                
            st.caption("💡 Mẹo: Nhấn chọn một dòng bất kỳ trong bảng dưới đây để xem lại ảnh, chi tiết và tùy chọn XÓA!" if lang=='vi' else "💡 Tip: Click any row below to view the image, details, and DELETE option!")
            
            display_history = filtered_df[['scan_time', 'trash_type', 'category', 'confidence']].copy()
            if lang == 'en':
                cat_map = {"Tái Chế": "Recyclable", "Hữu Cơ": "Organic", "Nguy Hại": "Hazardous", "Vô Cơ": "Non-recyclable"}
                display_history['category'] = display_history['category'].map(cat_map).fillna(display_history['category'])
                display_history.columns = ['Time', 'Waste Type', 'Category', 'Confidence (%)']
            else:
                display_history.columns = ['Thời gian', 'Loại rác', 'Phân loại', 'Độ tin cậy (%)']
            
            event = st.dataframe(
                display_history, 
                use_container_width=True, 
                height=300, 
                on_select="rerun", 
                selection_mode="single-row",
                hide_index=True 
            )
            
            selected_rows = event.selection.rows
            if selected_rows:
                index = selected_rows[0]
                selected_data = filtered_df.iloc[index] 
                
                st.markdown("#### 🔍 Chi tiết bản ghi được chọn" if lang=='vi' else "#### 🔍 Selected Record Details")
                col_img, col_txt = st.columns([1, 1.5])
                
                path = selected_data.get('image_path')
                
                with col_img:
                    if pd.notna(path) and os.path.exists(path):
                        st.image(Image.open(path), caption="Ảnh thực tế" if lang=='vi' else "Real Image", use_container_width=True)
                    else:
                        st.warning("Ảnh này không tồn tại trên hệ thống." if lang=='vi' else "This image does not exist on the system.")
                
                with col_txt:
                    cat_disp = selected_data['category']
                    if lang == 'en':
                        cat_disp = cat_map.get(cat_disp, cat_disp)
                        
                    st.write(f"📅 **{'Thời gian' if lang=='vi' else 'Time'}:** {selected_data['scan_time']}")
                    st.write(f"🤖 **{'Hệ thống AI dự đoán' if lang=='vi' else 'AI Prediction'}:** {selected_data['trash_type'].upper()}")
                    st.write(f"🗑️ **{'Phân loại' if lang=='vi' else 'Category'}:** {cat_disp}")
                    st.success(f"{'Độ tự tin' if lang=='vi' else 'Confidence'}: {selected_data['confidence']:.2f}%")
                    
                    if st.button("🗑️ Xóa bản ghi này" if lang=='vi' else "🗑️ Delete this record", type="primary"):
                        try:
                            if pd.notna(path) and os.path.exists(path):
                                os.remove(path)
                            record_id = int(selected_data['id'])
                            delete_history_and_feedback(record_id, path)
                            st.success("Đã xóa bản ghi thành công!" if lang=='vi' else "Record deleted successfully!")
                            st.rerun() 
                        except Exception as e:
                            st.error(f"Lỗi khi xóa: {e}" if lang=='vi' else f"Error deleting: {e}")

        st.markdown("---")
        st.subheader("📝 Lịch sử phản hồi của bạn" if lang=='vi' else "📝 Your Feedback History")

        if not df_feedback.empty:
            display_fb = df_feedback[['timestamp', 'predicted_class', 'true_class', 'is_correct']].copy()
            if lang == 'en':
                display_fb.columns = ['Time', 'Predicted', 'Actual', 'Is Correct?']
            else:
                display_fb.columns = ['Thời gian', 'Dự đoán', 'Thực tế', 'Đúng?']
            st.dataframe(display_fb, use_container_width=True, height=250, hide_index=True)
        else:
            st.info("Chưa có phản hồi nào được ghi nhận." if lang=='vi' else "No feedback recorded yet.")
            
        if not df.empty:
            st.markdown("---")
            st.subheader("🔴 Xóa dữ liệu" if lang=='vi' else "🔴 Delete Data")
            
            btn_txt = "XÓA TOÀN BỘ DỮ LIỆU CỦA TÔI" if lang=='vi' else "DELETE ALL MY DATA"
            if st.button(btn_txt, type="primary", use_container_width=True):
                try:
                    for path in df['image_path'].dropna():
                        if path and os.path.exists(path):
                            os.remove(path)
                    
                    delete_all_user_data(current_user)
                    
                    st.success("Đã dọn dẹp sạch sẽ toàn bộ dữ liệu! Đang tải lại..." if lang=='vi' else "All data completely purged! Reloading...")
                    st.rerun()
                except Exception as e:
                    st.error(f"Lỗi khi xóa dữ liệu: {e}" if lang=='vi' else f"Error deleting data: {e}")
                    
            warn_txt = "⚠️ Cảnh báo: Thao tác này sẽ xóa vĩnh viễn toàn bộ lịch sử quét ảnh và lịch sử phản hồi của bạn. Không thể hoàn tác!" if lang=='vi' else "⚠️ Warning: This action will permanently delete all your scanned history and feedback. Cannot be undone!"
            st.warning(warn_txt)

    except Exception as e:
        st.error(f"Đã xảy ra lỗi khi tải dữ liệu: {e}" if lang=='vi' else f"Error loading data: {e}")