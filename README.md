# Nghiên cứu, đánh giá và triển khai mô hình Deep Learning tối ưu cho hệ thống phân loại rác thải

## Giới thiệu dự án

Dự án nghiên cứu và so sánh các mô hình Deep Learning (VGG16, ResNet50, EfficientNet-B0) cho bài toán phân loại rác thải. Chọn mô hình tối ưu để triển khai ứng dụng web, cho phép người dùng tải ảnh rác thải lên và nhận kết quả phân loại kèm hướng dẫn xử lý.

**Tính năng chính:**
- Nhận dạng tự động 10 loại rác thải
- Hiển thị bản đồ Grad-CAM giải thích quyết định của model
- Tư vấn cách xử lý rác từ Gemini Vision AI
- Theo dõi lịch sử phân loại và thống kê
- Thu thập phản hồi người dùng để cải thiện

---

## Hướng dẫn cài đặt và chạy

### 1. Clone repository
```bash
git clone https://github.com/NVGBAOgit/waste-classification-system.git
cd waste-classification-system
```

### 2. Cài đặt thư viện
```bash
pip install -r SourceCode/requirements.txt
```

### 3. Tải dataset

**Phase 1 - So sánh (tùy chọn):**
Tải dataset **TrashNet** từ GitHub:
https://github.com/garythung/trashnet

Giải nén và đặt 6 thư mục (cardboard, glass, metal, paper, plastic, trash) vào `SourceCode/model_comparison/data/`

**Phase 2 - Fine-tune (bắt buộc):**
Tải bộ dữ liệu **Garbage Classification** (11 lớp ban đầu) từ Kaggle:
https://www.kaggle.com/datasets/nishchalkansara/garbage-classification

Giải nén, lấy các thư mục chính rồi đặt vào `SourceCode/data/`

**Lưu ý xử lý dataset:**
Dataset gốc có 11 loại. Đã gộp 2 loại tương tự (biological + organic) thành 1 loại **organic**. Một số ảnh cũng đã bị xóa để cân bằng dữ liệu giữa các loại. Nếu muốn dùng dataset gốc, hãy tự chỉnh sửa hoặc sử dụng tất cả 11 loại và cập nhật `CLASS_NAMES` trong code.

**Số lượng ảnh mỗi loại (sau xử lý - 10 lớp):**
- battery: 504 | cardboard: 403 | clothes: 504 | glass: 501
- metal: 410 | organic: 513 | paper: 504 | plastic: 482
- shoes: 504 | trash: 137

### 4. Setup API key Gemini
Tạo file `.env` tại `SourceCode/`:
```bash
GEMINI_API_KEY=your_api_key_here
```

Lấy API key từ: https://ai.google.dev/

### 5. Train mô hình (Fine-tune Phase 2)
```bash
cd SourceCode/fine_tune_efficientnet
python train.py
```

Lệnh này sẽ tạo file `.onnx` cần thiết cho API.

### 6. Chạy Backend API
```bash
cd SourceCode
python -m uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

### 7. Chạy Frontend Streamlit
Mở terminal khác:
```bash
cd SourceCode/app
streamlit run app.py
```

Ứng dụng sẽ mở tại: http://localhost:8501
