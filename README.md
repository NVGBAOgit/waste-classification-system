# Nghiên cứu, Đánh giá và Triển khai Mô hình Deep Learning Tối ưu cho Hệ thống Phân loại Rác Thải

## Giới thiệu dự án
Đề tài nghiên cứu và so sánh hiệu năng của các mô hình Deep Learning (VGG16, ResNet50, EfficientNet) trên bài toán phân loại rác thải. Từ kết quả đánh giá, lựa chọn mô hình tối ưu để triển khai ứng dụng web cho phép người dùng upload ảnh rác thải và nhận kết quả phân loại kèm hướng dẫn xử lý phù hợp.

## Hướng dẫn cài đặt và chạy chương trình
### 1. Cài đặt thư viện
pip install -r SourceCode/requirements.txt
### 2. Tải dataset 
Tải dataset TrashNet tại: https://github.com/garythung/trashnet (Dùng để so sánh 3 models)  
Giải nén và đặt 6 thư mục (cardboard, glass, metal, paper, plastic, trash) vào thư mục SourceCode/data/

Tải bộ dữ liệu **Garbage Classification** (10 lớp rác) từ Kaggle:  
🔗 [https://www.kaggle.com/datasets/nishchalkansara/garbage-classification](https://www.kaggle.com/datasets/nishchalkansara/garbage-classification) (Dùng chính để chạy dự án)

Sau khi tải về, giải nén và **chỉ lấy 10 thư mục con** sau:
- `battery`
- `cardboard`
- `clothes`
- `glass`
- `metal`
- `organic`
- `paper`
- `plastic`
- `shoes`
- `trash`

*(Mỗi lớp khuyến nghị từ 400 - 500 ảnh, hệ thống đã có Class Weights để tự cân bằng nếu thiếu).*  
Đặt các thư mục này vào thư mục `SourceCode/data/`.

