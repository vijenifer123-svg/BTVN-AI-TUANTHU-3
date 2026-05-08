import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
import os
import base64
import json 

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="VQH 3I BPF - Định giá xe máy", page_icon="🏍️", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #FAFAFA; }
    .block-container {
        padding-top: 0rem !important; 
        padding-bottom: 2rem !important;
    }
    [data-testid="stHeader"] { background: rgba(0,0,0,0); }
    
    div[data-testid="stButton"] > button[kind="primary"] {
        background-color: #FF5722; 
        color: white; border: none; border-radius: 8px;
        height: 45px; font-weight: bold; font-size: 16px; width: 100%; margin-bottom: 5px;
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover { background-color: #E64A19; color: white; }
    
    div[data-testid="stButton"] > button[kind="secondary"] {
        background-color: white; color: #FF5722; border: 1px solid #FF5722;
        border-radius: 8px; height: 45px; font-weight: bold; font-size: 16px; width: 100%;
    }
    div[data-testid="stButton"] > button[kind="secondary"]:hover { border-color: #E64A19; color: #E64A19; }
    
    div[data-testid="stDownloadButton"] > button {
        background-color: white; color: #4CAF50; border: 2px solid #4CAF50;
        border-radius: 8px; height: 45px; font-weight: bold; font-size: 16px; width: 100%; margin-bottom: 5px;
    }
    div[data-testid="stDownloadButton"] > button:hover { background-color: #4CAF50; color: white; border-color: #4CAF50;}
    
    .dashboard-card {
        background: white; padding: 20px; border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin-bottom: 20px; border: 1px solid #f0f0f0;
    }
    
    .history-item { border-bottom: 1px solid #eee; padding: 12px 0; }
    .history-item:last-child { border-bottom: none; }
    .history-title { font-weight: bold; color: #333; font-size: 15px;}
    .history-price { color: #FF5722; font-weight: bold; float: right; font-size: 15px;}
    .history-specs { color: gray; font-size: 13px; margin-top: 4px;}
    
    .top-navbar {
        background-color: #FFF3E0; 
        border-bottom: 2px solid #FF5722; 
        padding: 15px 30px; 
        margin-bottom: 35px;
        border-radius: 0 0 10px 10px; 
        display: flex; justify-content: space-between; align-items: center;
    }
    </style>
    """, unsafe_allow_html=True)

# --- QUẢN LÝ CHUYỂN TRANG & BỘ NHỚ LỊCH SỬ ---
if 'page' not in st.session_state: st.session_state.page = 'auth'
if 'selected_brand' not in st.session_state: st.session_state.selected_brand = 'Honda'
if 'current_prediction' not in st.session_state: st.session_state.current_prediction = None
if 'show_all_history' not in st.session_state: st.session_state.show_all_history = False
if 'auth_mode' not in st.session_state: st.session_state.auth_mode = 'login'

# --- TÍNH NĂNG ĐỌC FILE LƯU TÀI KHOẢN ---
if 'users' not in st.session_state:
    if os.path.exists("Backup_Users.json"):
        with open("Backup_Users.json", "r", encoding="utf-8") as f:
            st.session_state.users = json.load(f)
    else:
        st.session_state.users = {} 

# --- TÍNH NĂNG ĐỌC FILE LƯU LỊCH SỬ ---
if 'history' not in st.session_state:
    backup_file = "Backup_Lich_su_VQH_BPF.csv"
    if os.path.exists(backup_file):
        try:
            hist_df = pd.read_csv(backup_file)
            loaded_hist = []
            for _, row in hist_df.iterrows():
                loaded_hist.append({
                    "name": str(row.get("Tên xe", "")),
                    "specs": str(row.get("Thông số kỹ thuật", "")),
                    "price": str(row.get("Mức giá AI dự đoán", ""))
                })
            st.session_state.history = loaded_hist
        except:
            st.session_state.history = []
    else:
        st.session_state.history = [] 

def change_page(page_name):
    st.session_state.page = page_name

def get_logo_html(height="55px"):
    logo_path = "LOGO XE CŨ.jpg"
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        return f"<img src='data:image/jpeg;base64,{encoded_string}' style='height: {height}; object-fit: contain; mix-blend-mode: multiply;'>"
    return ""

def render_navbar():
    logo_html = get_logo_html(height="45px")
    st.markdown(f"""
        <div class="top-navbar">
            <div style="display: flex; align-items: center; gap: 15px;">
                <div>{logo_html}</div>
                <h2 style="color: #FF5722; margin: 0; font-weight: 900; font-size: 28px;">VQH 3I BPF</h2>
            </div>
            <div style="color: #666; font-weight: 500; font-size: 16px;">Hệ thống định giá xe thông minh</div>
        </div>
    """, unsafe_allow_html=True)

# --- BƯỚC 1: LOAD DỮ LIỆU & HUẤN LUYỆN CATBOOST ---
@st.cache_resource
def train_catboost_model():
    sheet_id = "1J1DzeSpK79bbPjCOAncytf2ByxFagslnl58NsTCPIp0"
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    df = pd.read_csv(url)
    
    all_cols = df.columns.tolist()
    col_gia = next((c for c in all_cols if 'giá' in c.lower()), 'Giá bán')
    col_hang = next((c for c in all_cols if 'hãng' in c.lower()), None)
    col_dong = next((c for c in all_cols if 'dòng' in c.lower()), None)
    col_nam = next((c for c in all_cols if 'năm' in c.lower()), None)
    col_km = next((c for c in all_cols if 'km' in c.lower()), None)
    col_tt = next((c for c in all_cols if 'tình' in c.lower()), None)
    col_pt = next((c for c in all_cols if 'phụ' in c.lower() or 'thay' in c.lower()), None)
    col_kv = next((c for c in all_cols if 'khu' in c.lower()), None)

    for col in [col_gia, col_km]:
        if col in df.columns:
            temp = df[col].astype(str)
            temp = temp.str.replace(r'[\.,]\d{1,2}$', '', regex=True) 
            temp = temp.str.replace(r'[^\d]', '', regex=True)
            df[col] = pd.to_numeric(temp, errors='coerce')
            
    if col_tt in df.columns:
        df[col_tt] = pd.to_numeric(df[col_tt].astype(str).str.replace(',', '.'), errors='coerce')

    df = df.dropna()

    feature_cols = [c for c in [col_hang, col_dong, col_nam, col_km, col_tt, col_pt, col_kv] if c is not None]
    X = df[feature_cols]
    y = df[col_gia]
    
    cat_features = X.select_dtypes(include=['object', 'string']).columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    model = CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=8, loss_function='RMSE', verbose=False, random_seed=42)
    model.fit(X_train, y_train, cat_features=cat_features)
    
    return model, df, feature_cols, col_hang, col_dong, col_nam, col_km, col_tt, col_pt, col_kv

try:
    model, dataset, feature_names, col_hang, col_dong, col_nam, col_km, col_tt, col_pt, col_kv = train_catboost_model()
    data_loaded = True
except Exception as e:
    st.error(f"Lỗi khởi tạo dữ liệu: {e}")
    data_loaded = False

if data_loaded:
    
    def get_image(brand_name):
        b = str(brand_name).lower()
        if 'honda' in b: return 'https://images.unsplash.com/photo-1558981403-c5f9899a28bc?w=500&auto=format&fit=crop'
        elif 'yamaha' in b: return 'https://images.unsplash.com/photo-1568772585407-9361f9bf3a87?w=500&auto=format&fit=crop'
        elif 'suzuki' in b: return 'https://images.unsplash.com/photo-1590403332410-b96e5d8b7a6d?w=500&auto=format&fit=crop'
        elif 'piaggio' in b: return 'https://images.unsplash.com/photo-1620916297397-a4a5402a3c6c?w=500&auto=format&fit=crop'
        else: return 'https://images.unsplash.com/photo-1449426468159-d96dbf08f19f?w=500&auto=format&fit=crop'

    # ==========================================
    # 1. TRANG ĐĂNG NHẬP / ĐĂNG KÝ
    # ==========================================
    if st.session_state.page == 'auth':
        render_navbar() 
        
        col_form, col_space, col_image = st.columns([1.3, 0.2, 1.8])
        
        with col_form:
            st.write("<br><br>", unsafe_allow_html=True)
            
            logo_html = get_logo_html(height="65px")
            st.markdown(f"""
                <div style='display: flex; align-items: center; gap: 15px; margin-bottom: 5px;'>
                    <div>{logo_html}</div>
                    <h1 style='color:#FF5722; font-weight:900; margin:0; font-size: 38px;'>VQH 3I BPF</h1>
                </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.auth_mode == 'login':
                # ====== FORM ĐĂNG NHẬP (ĐÃ GỌT DŨA GỌN GÀNG) ======
                st.markdown("<h3 style='color:#333; margin-top: 10px; margin-bottom: 20px;'>Đăng nhập</h3>", unsafe_allow_html=True)
                
                email_input = st.text_input("Nhập email hoặc số điện thoại *", key="login_email")
                pass_input = st.text_input("Nhập mật khẩu *", type="password", key="login_pass")
                
                st.write("<br>", unsafe_allow_html=True)
                if st.button("Đăng nhập", type="primary", use_container_width=True):
                    if email_input.strip() == "" or pass_input.strip() == "":
                        st.error("⚠️ Vui lòng nhập đầy đủ Email/SĐT và Mật khẩu!")
                    else:
                        # Tự động lưu tài khoản vào hệ thống ngầm
                        if email_input not in st.session_state.users:
                            st.session_state.users[email_input] = pass_input
                            with open("Backup_Users.json", "w", encoding="utf-8") as f:
                                json.dump(st.session_state.users, f)
                        change_page('dashboard')
                        st.rerun()
                
                st.write("<br>", unsafe_allow_html=True)
                st.markdown("<div style='text-align:center; color: gray; margin-bottom: 5px;'>Chưa có tài khoản?</div>", unsafe_allow_html=True)
                if st.button("Đăng ký ngay", type="secondary", use_container_width=True):
                    st.session_state.auth_mode = 'register'
                    st.rerun()

            else:
                # ====== FORM ĐĂNG KÝ (ĐÃ BỎ Ô HỌ VÀ TÊN) ======
                st.markdown("<h3 style='color:#333; margin-top: 10px; margin-bottom: 20px;'>Đăng ký tài khoản</h3>", unsafe_allow_html=True)
                
                phone_reg = st.text_input("Nhập email hoặc số điện thoại *", key="reg_email")
                pass_reg = st.text_input("Nhập mật khẩu *", type="password", key="reg_pass")
                agree = st.checkbox("Tôi đồng ý với Điều khoản sử dụng và Chính sách bảo mật của VQH 3I BPF.")
                
                st.write("<br>", unsafe_allow_html=True)
                if st.button("Đăng ký", type="primary", use_container_width=True):
                    if phone_reg.strip() == "" or pass_reg.strip() == "":
                        st.error("⚠️ Vui lòng điền đầy đủ các thông tin bắt buộc!")
                    elif not agree:
                        st.error("⚠️ Bạn cần đồng ý với Điều khoản sử dụng để tiếp tục.")
                    else:
                        st.session_state.users[phone_reg] = pass_reg
                        with open("Backup_Users.json", "w", encoding="utf-8") as f:
                            json.dump(st.session_state.users, f)
                        st.success("✅ Đăng ký thành công! Hệ thống đang tự động đăng nhập...")
                        st.session_state.auth_mode = 'login' 
                        change_page('dashboard')
                        st.rerun()
                        
                st.write("<br>", unsafe_allow_html=True)
                st.markdown("<div style='text-align:center; color: gray; margin-bottom: 5px;'>Đã có tài khoản?</div>", unsafe_allow_html=True)
                if st.button("Quay lại Đăng nhập", type="secondary", use_container_width=True):
                    st.session_state.auth_mode = 'login'
                    st.rerun()

        with col_image:
            st.markdown("""
                <div style="height: 85vh; background-image: url('https://images.unsplash.com/photo-1558981806-ec527fa84c39?q=80&w=1000'); background-size: cover; background-position: center; border-radius: 20px;">
                </div>
            """, unsafe_allow_html=True)

    # ==========================================
    # 2. TRANG DASHBOARD CHÍNH 
    # ==========================================
    elif st.session_state.page == 'dashboard':
        render_navbar()
        col_main, col_sidebar = st.columns([1.5, 1])
        
        with col_main:
            header_col1, header_col2 = st.columns([3, 1])
            with header_col1:
                st.markdown("<h3 style='margin-bottom:0;'>Định giá xe máy cũ ⚡</h3>", unsafe_allow_html=True)
                st.markdown("<p style='color:gray;'>Chọn hãng xe để bắt đầu ⬇️</p>", unsafe_allow_html=True)
            with header_col2:
                if st.button("Tài khoản", type="secondary", use_container_width=True):
                    change_page('account')
                    st.rerun()
            
            st.write("---")
            
            brand_logos = {
                "Honda": "image_5582f5.png",
                "Yamaha": "image_557baf.png",
                "Suzuki": "image_557ef8.png",
                "Kymco": "image_557f52.png",
                "Kawasaki": "image_557af9.png"
            }
            
            cols = st.columns(5)
            for i, (brand, img_path) in enumerate(brand_logos.items()):
                with cols[i]:
                    try:
                        st.image(img_path, use_container_width=True)
                    except:
                        st.error(f"Thiếu {brand}")
                    
                    if st.button(f"Chọn {brand}", key=f"btn_{brand}", type="secondary", use_container_width=True):
                        st.session_state.selected_brand = brand
                        change_page('predict')
                        st.rerun()
            
            st.write("<br>", unsafe_allow_html=True)
            st.markdown("<h4 style='color:#FF5722;'>🕒 Lịch sử định giá gần đây</h4>", unsafe_allow_html=True)
            st.write("---")
            
            history_list = st.session_state.history
            display_limit = 4
            items_to_show = history_list if st.session_state.show_all_history else history_list[:display_limit]
            
            if len(items_to_show) == 0:
                st.markdown("<p style='color:gray; font-style:italic;'>Chưa có lịch sử định giá nào.</p>", unsafe_allow_html=True)
            
            for item in items_to_show:
                st.markdown(f"""
                    <div class='history-item'>
                        <span class='history-title'>{item['name']}</span>
                        <span class='history-price'>{item['price']}</span>
                        <div class='history-specs'>{item['specs']}</div>
                    </div>
                """, unsafe_allow_html=True)
                
            if len(history_list) > display_limit:
                st.write("<br>", unsafe_allow_html=True)
                if not st.session_state.show_all_history:
                    if st.button("Xem tất cả lịch sử", type="secondary", use_container_width=True):
                        st.session_state.show_all_history = True
                        st.rerun()
                else:
                    if st.button("Thu gọn danh sách", type="secondary", use_container_width=True):
                        st.session_state.show_all_history = False
                        st.rerun()
            
        with col_sidebar:
            if st.button("Định giá ngay", type="primary", use_container_width=True):
                st.session_state.selected_brand = 'Honda' 
                change_page('predict')
                st.rerun()
                
            st.write("---")
            st.markdown("#### 🔥 Xe nổi bật")
            st.image('https://images.unsplash.com/photo-1558981403-c5f9899a28bc?w=500', use_container_width=True)
            st.markdown("**Honda SH 150i 2023**<br><span style='color:#FF5722;font-weight:bold; font-size:18px;'>85.000.000 đ</span>", unsafe_allow_html=True)
            st.write("---")
            st.image('https://images.unsplash.com/photo-1568772585407-9361f9bf3a87?w=500', use_container_width=True)
            st.markdown("**Yamaha NVX 155 VVA**<br><span style='color:#FF5722;font-weight:bold; font-size:18px;'>45.500.000 đ</span>", unsafe_allow_html=True)

    # ==========================================
    # 3. TRANG ĐỊNH GIÁ (FORM NHẬP LIỆU)
    # ==========================================
    elif st.session_state.page == 'predict':
        render_navbar()
        col_back, col_title, col_empty = st.columns([1, 4, 1])
        if col_back.button("⬅️ Trở về", type="secondary"): 
            change_page('dashboard')
            st.rerun()
            
        col_left, col_form, col_right = st.columns([1.2, 2.6, 1.2])
        
        with col_left:
            st.write("<br><br>", unsafe_allow_html=True)
            st.markdown("""
                <div class='dashboard-card'>
                    <h4 style='color: #FF5722;'>💡 Mẹo định giá</h4>
                    <p style='font-size: 14px; color: #555;'>Để AI tính toán chuẩn xác nhất:</p>
                    <ul style='font-size: 13px; color: #666; padding-left: 20px;'>
                        <li style='margin-bottom: 5px;'><b>Số KM:</b> Nhập đúng ODO hiện tại.</li>
                        <li style='margin-bottom: 5px;'><b>Tình trạng:</b> Xước nhẹ chọn 7-8, như mới chọn 9-10.</li>
                        <li><b>Phụ tùng:</b> Xe nguyên bản (zin) luôn giữ giá tốt hơn.</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)

        with col_form:
            st.markdown("<h2 style='text-align:center; color:#FF5722;'>Định giá xe máy cũ</h2>", unsafe_allow_html=True)
            st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
            st.markdown("#### 📋 Thông tin xe của bạn", unsafe_allow_html=True)
            st.write("*(Bạn có thể click vào ô và gõ chữ để tìm kiếm nhanh)*")
            st.write("<br>", unsafe_allow_html=True)
            
            all_brands_list = list(dataset[col_hang].dropna().unique()) if col_hang else ["Honda", "Yamaha"]
            default_idx = all_brands_list.index(st.session_state.selected_brand) if st.session_state.selected_brand in all_brands_list else 0

            h = st.selectbox("1️⃣ Hãng xe", all_brands_list, index=default_idx)
            dong_xe_list = dataset[dataset[col_hang]==h][col_dong].dropna().unique() if col_dong else []
            d = st.selectbox("2️⃣ Dòng xe", dong_xe_list)
            
            km = st.number_input("3️⃣ Số KM đã chạy", min_value=0, max_value=500000, value=15000, step=1)
            n = st.number_input("4️⃣ Năm sản xuất", min_value=1990, max_value=2026, value=2021, step=1)
            tt = st.slider("5️⃣ Tình trạng xe (1 Tệ - 10 Hoàn hảo)", min_value=1.0, max_value=10.0, value=8.0, step=0.5)
            
            pt_options = list(dataset[col_pt].dropna().unique()) if col_pt else ["Không", "Có"]
            pt = st.radio("6️⃣ Thay phụ tùng chưa?", pt_options, horizontal=True)
            
            kv_options = list(dataset[col_kv].dropna().unique()) if col_kv else ["TP Hồ Chí Minh", "Hà Nội", "Đà Nẵng"]
            kv = st.selectbox("7️⃣ Khu vực", kv_options)
            
            st.write("<br>", unsafe_allow_html=True)
            if st.button("🚀 Tiếp theo (Tính giá dự đoán)", type="primary", use_container_width=True):
                input_dict = {}
                if col_hang: input_dict[col_hang] = h
                if col_dong: input_dict[col_dong] = d
                if col_nam: input_dict[col_nam] = n
                if col_km: input_dict[col_km] = km
                if col_tt: input_dict[col_tt] = tt
                if col_pt: input_dict[col_pt] = pt
                if col_kv: input_dict[col_kv] = kv
                
                input_data = pd.DataFrame([input_dict])[feature_names]
                pred = model.predict(input_data)[0]
                pred = max(pred, 0)
                
                pt_status = "Đã thay" if pt == "Có" or pt == "đã thay" else "Không thay"
                new_history_item = {
                    "name": f"{h} {d} {int(n)}",
                    "specs": f"{km:,.0f} km • {pt_status} PT • {kv}",
                    "price": f"{pred:,.0f} đ"
                }
                st.session_state.history.insert(0, new_history_item)
                
                hist_df = pd.DataFrame(st.session_state.history)
                hist_df.columns = ["Tên xe", "Thông số kỹ thuật", "Mức giá AI dự đoán"]
                try:
                    hist_df.to_csv("Backup_Lich_su_VQH_BPF.csv", index=False, encoding='utf-8-sig')
                except:
                    pass
                
                st.session_state.current_prediction = {
                    'h': h, 'd': d, 'n': n, 'km': km, 'tt': tt, 'pt': pt, 'kv': kv, 'price': pred
                }
                change_page('result')
                st.rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)

        with col_right:
            st.write("<br><br>", unsafe_allow_html=True)
            st.markdown(f"""
                <div class='dashboard-card'>
                    <h4 style='color: #FF5722;'>🛡️ Công nghệ AI</h4>
                    <img src="{get_image(h)}" style="width: 100%; border-radius: 8px; margin-bottom: 10px;">
                    <p style='font-size: 13px; color: gray; margin-top: 10px;'>Phân tích hàng ngàn mẫu dữ liệu giao dịch thực tế trên thị trường để loại bỏ cảm tính, đưa ra mức định giá sát nhất.</p>
                </div>
            """, unsafe_allow_html=True)

    # ==========================================
    # 3.5 TRANG KẾT QUẢ ĐỊNH GIÁ 
    # ==========================================
    elif st.session_state.page == 'result':
        render_navbar()
        curr = st.session_state.current_prediction
        
        st.markdown(f"<h2 style='color:#333; margin-bottom: 20px;'>Kết quả định giá cho {curr['h']} {curr['d']} {int(curr['n'])}</h2>", unsafe_allow_html=True)
        
        col_info, col_price = st.columns([1.5, 2.5])
        
        with col_info:
            st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
            st.markdown("<h4 style='border-bottom: 1px solid #eee; padding-bottom: 10px; color:#FF5722;'>Thông Tin Xe Của Bạn</h4>", unsafe_allow_html=True)
            
            st.markdown(f"""
                <p style="margin-bottom:8px;"><b>Hãng xe:</b> <span style="float:right;">{curr['h']}</span></p>
                <p style="margin-bottom:8px;"><b>Dòng xe:</b> <span style="float:right;">{curr['d']}</span></p>
                <p style="margin-bottom:8px;"><b>Năm sản xuất:</b> <span style="float:right;">{int(curr['n'])}</span></p>
                <p style="margin-bottom:8px;"><b>Số KM đã đi:</b> <span style="float:right;">{curr['km']:,.0f} km</span></p>
                <p style="margin-bottom:8px;"><b>Tình trạng:</b> <span style="float:right;">{curr['tt']} / 10</span></p>
                <p style="margin-bottom:8px;"><b>Phụ tùng:</b> <span style="float:right;">{curr['pt']}</span></p>
                <p style="margin-bottom:8px; border-bottom: 1px solid #eee; padding-bottom: 10px;"><b>Khu vực:</b> <span style="float:right;">{curr['kv']}</span></p>
            """, unsafe_allow_html=True)
            
            st.write("<br>", unsafe_allow_html=True)
            
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("🔄 Định giá xe khác", type="secondary", use_container_width=True):
                    change_page('predict')
                    st.rerun()
            with btn_col2:
                if st.button("🏠 Quay lại trang chủ", type="primary", use_container_width=True):
                    change_page('dashboard')
                    st.rerun()
                
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_price:
            st.markdown("<div class='dashboard-card' style='text-align: center;'>", unsafe_allow_html=True)
            st.markdown("<h4 style='color: #666;'>Kết quả định giá xe của bạn</h4>", unsafe_allow_html=True)
            
            price = curr['price']
            lower_price = price * 0.96
            upper_price = price * 1.04
            
            st.markdown(f"""
                <div style='background-color:#FFF3E0; border:2px dashed #FF5722; padding:40px; border-radius:12px; margin-top:15px;'>
                    <h1 style='color: #FF5722; margin: 0; font-size: 50px;'>{lower_price:,.0f} - {upper_price:,.0f} VNĐ</h1>
                    <p style='color: #E64A19; margin-top: 15px; font-weight: bold; font-size:18px;'>Giá ước tính chuẩn: {price:,.0f} VNĐ</p>
                    <p style='font-size: 14px; color: gray; margin-top: 15px;'>Giá ước tính dựa trên AI CatBoost và dữ liệu thị trường thực tế. Với mức giá này, xe sẽ dễ dàng thanh khoản trong khoảng 7-14 ngày.</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.write("<br>", unsafe_allow_html=True)
            st.markdown("🔹 *Kết quả định giá chi tiết đã được tự động lưu vào Lịch sử.*")
            st.markdown("</div>", unsafe_allow_html=True)

    # ==========================================
    # 4. TRANG TÀI KHOẢN (ĐĂNG XUẤT)
    # ==========================================
    elif st.session_state.page == 'account':
        render_navbar()
        col1, col2, col3 = st.columns([1, 1.5, 1]) 
        with col2:
            st.write("<br><br><br>", unsafe_allow_html=True)
            
            logo_html = get_logo_html(height="120px")
            st.markdown(f"""
                <div style='display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 15px; margin-bottom: 20px;'>
                    {logo_html}
                    <h1 style='color:#FF5722; font-weight:900; margin:0; font-size: 45px;'>VQH 3I BPF</h1>
                </div>
            """, unsafe_allow_html=True)
            
            hist_df = pd.DataFrame(st.session_state.history)
            if not hist_df.empty:
                hist_df.columns = ["Tên xe", "Thông số kỹ thuật", "Mức giá AI dự đoán"]
                csv = hist_df.to_csv(index=False).encode('utf-8-sig') 
                st.download_button(
                    label="📥 Tải xuống Lịch sử định giá",
                    data=csv,
                    file_name="Lich_su_VQH_BPF.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            if st.button("Đăng xuất", type="primary", use_container_width=True):
                try:
                    hist_df.to_csv("Backup_Lich_su_VQH_BPF.csv", index=False, encoding='utf-8-sig')
                except:
                    pass
                    
                change_page('auth')
                st.session_state.auth_mode = 'login' 
                st.rerun()
            
            if st.button("Quay lại trang chủ", type="secondary", use_container_width=True):
                change_page('dashboard')
                st.rerun()
