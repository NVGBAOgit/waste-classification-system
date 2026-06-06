import streamlit as st
import database
import page_scanner
import page_dashboard

# CẤU HÌNH TRANG 
st.set_page_config(
    page_title="GreenAI - Phân loại rác", 
    page_icon="♻️", 
    layout="wide"
)

# CUSTOM CSS
custom_css = """
    <style>
        input[type="password"]::-ms-reveal,
        input[type="password"]::-ms-clear { display: none !important; }
        input[type="password"]::-webkit-contacts-auto-fill-button,
        input[type="password"]::-webkit-credentials-auto-fill-button { display: none !important; }
        
        div.stButton > button:first-child {
            background-color: #16a34a;
            color: white;
            border-radius: 8px;
            border: none;
        }
        div.stButton > button:first-child:hover {
            background-color: #15803d;
        }
        
        div[data-testid="stAlert"] {
            border-radius: 10px;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        }
    </style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# KHỞI TẠO DATABASE
database.init_db()
database.auto_backup_db()

# KHỞI TẠO SESSION STATE 
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ""
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'scanner'

# MAIN LOGIC 
if not st.session_state['logged_in']:
    # TRANG ĐĂNG NHẬP / ĐĂNG KÝ
    st.title("🔐 Hệ thống Phân loại Rác GreenAI")
    st.markdown("Vui lòng đăng nhập hoặc tạo tài khoản để sử dụng hệ thống.")
    
    tab_login, tab_register = st.tabs(["Đăng nhập", "Đăng ký"])
    
    with tab_login:
        login_user = st.text_input("Tên đăng nhập", key="login_user")
        login_pass = st.text_input("Mật khẩu", type="password", key="login_pass")
        if st.button("Đăng nhập"):
            if database.login_user(login_user, login_pass):
                st.session_state['logged_in'] = True
                st.session_state['username'] = login_user
                st.rerun()
            else:
                st.error("Sai tên đăng nhập hoặc mật khẩu!")
                
    with tab_register:
        reg_user = st.text_input("Tên đăng nhập (mới)", key="reg_user")
        reg_pass = st.text_input("Mật khẩu", type="password", key="reg_pass")
        reg_pass_confirm = st.text_input("Nhập lại mật khẩu", type="password", key="reg_pass_confirm")
        if st.button("Đăng ký"):
            if reg_pass != reg_pass_confirm:
                st.error("Mật khẩu không khớp!")
            elif len(reg_user) < 3:
                st.error("Tên đăng nhập phải từ 3 ký tự trở lên.")
            else:
                if database.register_user(reg_user, reg_pass):
                    st.success("Đăng ký thành công!")
                else:
                    st.error("Tên đăng nhập đã tồn tại!")

else:
    # TRANG CHÍNH (SAU ĐĂNG NHẬP)
    st.sidebar.title("Quản lý hệ thống")
    st.sidebar.markdown(f"👤 Xin chào, **{st.session_state['username']}**")
    
    if st.sidebar.button("Đăng xuất", type="primary"):
        st.session_state.clear()
        st.session_state['logged_in'] = False
        st.rerun()
 
    st.sidebar.markdown("---")
    
    # Điều hướng trang
    st.sidebar.markdown("### 🧭 Điều hướng")
    page_idx = 0 if st.session_state['current_page'] == 'scanner' else 1
    menu = st.sidebar.radio(
        "", 
        ["🔍 Nhận diện Rác", "📊 Lịch sử Phân loại"], 
        index=page_idx, 
        label_visibility="collapsed"
    )
    
    # Cập nhật trang khi người dùng click menu
    if menu == "🔍 Nhận diện Rác" and st.session_state['current_page'] != 'scanner':
        st.session_state['current_page'] = 'scanner'
        st.rerun()
    elif menu == "📊 Lịch sử Phân loại" and st.session_state['current_page'] != 'dashboard':
        st.session_state['current_page'] = 'dashboard'
        st.rerun()
    
    # Hiển thị trang chức năng
    if st.session_state['current_page'] == 'scanner':
        page_scanner.show_scanner()
    elif st.session_state['current_page'] == 'dashboard':
        page_dashboard.show_dashboard()