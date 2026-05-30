import streamlit as st
import database
import page_scanner
import page_dashboard

# Thiết lập giao diện chung 
st.set_page_config(page_title="GreenAI - Phân loại rác", page_icon="♻️", layout="wide")

custom_css = """
    <style>
        /* Ẩn con mắt tự động của trình duyệt ở ô mật khẩu */
        input[type="password"]::-ms-reveal,
        input[type="password"]::-ms-clear { display: none !important; }
        input[type="password"]::-webkit-contacts-auto-fill-button,
        input[type="password"]::-webkit-credentials-auto-fill-button { display: none !important; }
        
        /* Đổi màu các nút bấm chính sang xanh lá */
        div.stButton > button:first-child {
            background-color: #16a34a;
            color: white;
            border-radius: 8px;
            border: none;
        }
        div.stButton > button:first-child:hover {
            background-color: #15803d;
        }
        
        /* Bo góc làm đẹp các hộp thông báo */
        div[data-testid="stAlert"] {
            border-radius: 10px;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        }
    </style>
    """
st.markdown(custom_css, unsafe_allow_html=True)

# Khởi tạo Database nếu chưa có
database.init_db()

# KÍCH HOẠT AUTO BACKUP KHI ỨNG DỤNG CHẠY
database.auto_backup_db()

# Cài đặt trạng thái Session
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ""
if 'lang' not in st.session_state:
    st.session_state['lang'] = 'vi'

if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'scanner'

lang = st.session_state['lang']

# ĐIỀU HƯỚNG TRẠNG THÁI
if not st.session_state['logged_in']:
    st.title("🔐 Hệ thống Phân loại Rác GreenAI" if lang == 'vi' else "🔐 GreenAI Waste Classification System")
    st.markdown("Vui lòng đăng nhập hoặc tạo tài khoản để sử dụng hệ thống." if lang == 'vi' else "Please login or create an account to use the system.")
    
    tab_login, tab_register = st.tabs(["Đăng nhập", "Đăng ký"] if lang == 'vi' else ["Login", "Register"])
    
    with tab_login:
        login_user_input = st.text_input("Tên đăng nhập" if lang == 'vi' else "Username", key="login_user")
        login_pass_input = st.text_input("Mật khẩu" if lang == 'vi' else "Password", type="password", key="login_pass")
        if st.button("Đăng nhập" if lang == 'vi' else "Login"):
            if database.login_user(login_user_input, login_pass_input):
                st.session_state['logged_in'] = True
                st.session_state['username'] = login_user_input
                st.rerun()
            else:
                st.error("Sai tên đăng nhập hoặc mật khẩu!" if lang == 'vi' else "Invalid username or password!")
                
    with tab_register:
        reg_user_input = st.text_input("Tên đăng nhập mới" if lang == 'vi' else "New Username", key="reg_user")
        reg_pass_input = st.text_input("Mật khẩu" if lang == 'vi' else "Password", type="password", key="reg_pass")
        reg_pass_confirm = st.text_input("Nhập lại mật khẩu" if lang == 'vi' else "Confirm Password", type="password", key="reg_pass_confirm")
        if st.button("Đăng ký" if lang == 'vi' else "Register"):
            if reg_pass_input != reg_pass_confirm:
                st.error("Mật khẩu không khớp!" if lang == 'vi' else "Passwords do not match!")
            elif len(reg_user_input) < 3:
                st.error("Tên đăng nhập phải từ 3 ký tự trở lên." if lang == 'vi' else "Username must be at least 3 characters.")
            else:
                if database.register_user(reg_user_input, reg_pass_input):
                    st.success("Đăng ký thành công!" if lang == 'vi' else "Registration successful!")
                else:
                    st.error("Tên đăng nhập đã tồn tại!" if lang == 'vi' else "Username already exists!")

else:
    # ĐỊNH NGHĨA TỪ ĐIỂN CHO SIDEBAR DỰA THEO NGÔN NGỮ
    if lang == 'vi':
        sys_title = "Quản lý hệ thống"
        hello_txt = f"👤 Xin chào, **{st.session_state['username']}**"
        logout_txt = "Đăng xuất"
        nav_title = "### 🧭 Điều hướng"
        menu_scan = "🔍 Nhận diện Rác"
        menu_dash = "📊 Lịch sử Phân loại"
        lang_title = "### 🌐 Ngôn ngữ"
        lang_opts = ["Tiếng Việt", "Tiếng Anh"]
    else:
        sys_title = "System Management"
        hello_txt = f"👤 Hello, **{st.session_state['username']}**"
        logout_txt = "Logout"
        nav_title = "### 🧭 Navigation"
        menu_scan = "🔍 Waste Scanner"
        menu_dash = "📊 Classification History"
        lang_title = "### 🌐 Language"
        lang_opts = ["Vietnamese", "English"]

    # SIDEBAR
    st.sidebar.title(sys_title)
    st.sidebar.markdown(hello_txt)
    
    if st.sidebar.button(logout_txt, type="primary"):
        saved_user = st.session_state['username']
        st.session_state.clear()
        st.session_state['logged_in'] = False
        st.session_state['username'] = ""
        st.session_state['login_user'] = saved_user 
        st.rerun()
 
    st.sidebar.markdown("---")
    
    # Khu vực Điều hướng 
    st.sidebar.markdown(nav_title)
    
    page_idx = 0 if st.session_state['current_page'] == 'scanner' else 1
    menu = st.sidebar.radio("", [menu_scan, menu_dash], index=page_idx, label_visibility="collapsed")
    
    # Cập nhật biến nhớ khi người dùng click sang trang khác
    if menu in ["🔍 Nhận diện Rác", "🔍 Waste Scanner"]:
        st.session_state['current_page'] = 'scanner'
    elif menu in ["📊 Lịch sử Phân loại", "📊 Classification History"]:
        st.session_state['current_page'] = 'dashboard'
    
    st.sidebar.markdown("---")
    
    # Khu vực Ngôn ngữ
    st.sidebar.markdown(lang_title)
    current_lang_idx = 0 if lang == 'vi' else 1
    selected_lang = st.sidebar.radio("", lang_opts, index=current_lang_idx, label_visibility="collapsed")
    
    # Bắt sự kiện đổi ngôn ngữ
    if selected_lang in ["Tiếng Anh", "English"] and lang == 'vi':
        st.session_state['lang'] = 'en'
        st.rerun()
    elif selected_lang in ["Tiếng Việt", "Vietnamese"] and lang == 'en':
        st.session_state['lang'] = 'vi'
        st.rerun()
    
    # GỌI CÁC TRANG CHỨC NĂNG
    if st.session_state['current_page'] == 'scanner':
        page_scanner.show_scanner()
    elif st.session_state['current_page'] == 'dashboard':
        page_dashboard.show_dashboard()