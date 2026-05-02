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

# Cài đặt trạng thái Session
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ""

# ĐIỀU HƯỚNG TRẠNG THÁI
if not st.session_state['logged_in']:
    st.title("🔐 Hệ thống Phân loại Rác GreenAI")
    st.markdown("Vui lòng đăng nhập hoặc tạo tài khoản để sử dụng hệ thống.")
    
    tab_login, tab_register = st.tabs(["Đăng nhập", "Đăng ký"])
    
    with tab_login:
        login_user_input = st.text_input("Tên đăng nhập", key="login_user")
        login_pass_input = st.text_input("Mật khẩu", type="password", key="login_pass")
        if st.button("Đăng nhập"):
            if database.login_user(login_user_input, login_pass_input):
                st.session_state['logged_in'] = True
                st.session_state['username'] = login_user_input
                st.rerun()
            else:
                st.error("Sai tên đăng nhập hoặc mật khẩu!")
                
    with tab_register:
        reg_user_input = st.text_input("Tên đăng nhập mới", key="reg_user")
        reg_pass_input = st.text_input("Mật khẩu", type="password", key="reg_pass")
        reg_pass_confirm = st.text_input("Nhập lại mật khẩu", type="password", key="reg_pass_confirm")
        if st.button("Đăng ký"):
            if reg_pass_input != reg_pass_confirm:
                st.error("Mật khẩu không khớp!")
            elif len(reg_user_input) < 3:
                st.error("Tên đăng nhập phải từ 3 ký tự trở lên.")
            else:
                if database.register_user(reg_user_input, reg_pass_input):
                    st.success("Đăng ký thành công!")
                else:
                    st.error("Tên đăng nhập đã tồn tại!")

else:
    # NẾU ĐÃ ĐĂNG NHẬP THÀNH CÔNG -> GỌI CÁC MODULE GIAO DIỆN RA
    st.sidebar.title("Quản lý hệ thống")
    st.sidebar.markdown(f"👤 Xin chào, **{st.session_state['username']}**")
    
    if st.sidebar.button("Đăng xuất"):
        saved_user = st.session_state['username']
        st.session_state.clear()
        st.session_state['logged_in'] = False
        st.session_state['username'] = ""
        st.session_state['login_user'] = saved_user 
        st.rerun()
 
    st.sidebar.markdown("---")
    menu = st.sidebar.radio("Điều hướng:", ["🔍 Nhận diện Rác", "📊 Lịch sử Phân loại"])
    
    if menu == "🔍 Nhận diện Rác":
        page_scanner.show_scanner()
    elif menu == "📊 Lịch sử Phân loại":
        page_dashboard.show_dashboard()