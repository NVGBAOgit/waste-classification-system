import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

def load_model(weights_path: str, num_classes: int = 10):
    """
    Khởi tạo mô hình EfficientNet-B0 và nạp trọng số đã train.
    """
    # Khởi tạo khung mô hình
    model = models.efficientnet_b0(weights=None)
    
    # Sửa lớp phân loại cuối cùng cho phù hợp với 10 loại rác
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    # Load trọng số từ file .pth (nạp lên CPU để chạy được trên mọi máy tính)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def generate_gradcam(model, image_path: str, save_path: str = "gradcam_output.jpg"):
    """
    Sinh bản đồ nhiệt Grad-CAM và lưu thành ảnh mới.
    """
    # Đích nhắm: Lớp tích chập cuối cùng của EfficientNet-B0
    # Nơi chứa thông tin không gian tốt nhất trước khi mô hình đưa ra quyết định
    target_layers = [model.features[-1]]
    
    # Khởi tạo công cụ Grad-CAM
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # Đọc và tiền xử lý ảnh đầu vào
    rgb_img = cv2.imread(image_path, 1)
    if rgb_img is None:
        raise FileNotFoundError(f"Không thể đọc được ảnh tại: {image_path}")
        
    rgb_img = rgb_img[:, :, ::-1] # Chuyển BGR sang RGB
    rgb_img = cv2.resize(rgb_img, (224, 224))       # Đưa về kích thước chuẩn của mạng
    rgb_img = np.float32(rgb_img) / 255             # Chuẩn hóa về khoảng [0, 1]
    
    # Chuẩn hóa theo tiêu chuẩn của ImageNet
    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    
    # Chạy mô hình để lấy Grad-CAM cho class dự đoán cao nhất (targets=None)
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)
    grayscale_cam = grayscale_cam[0, :]
    
    # Đè bản đồ nhiệt lên ảnh gốc
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    # Lưu ảnh kết quả
    cv2.imwrite(save_path, visualization[:, :, ::-1]) # Chuyển lại sang BGR để lưu bằng OpenCV
    return save_path

# ==========================================
# KHU VỰC CHẠY THỬ MÃ (TESTING)
# ==========================================
if __name__ == "__main__":
    import os
    
    # Tự động lấy đường dẫn tuyệt đối để tránh lỗi dù bạn chạy code từ thư mục nào
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Đường dẫn file trọng số và ảnh test (lùi ra ngoài thư mục gốc)
    MODEL_WEIGHTS = os.path.join(BASE_DIR, "saved_models", "efficientnet_finetune_best_v2.pth")
    TEST_IMAGE = os.path.join(BASE_DIR, "test_image.jpg")
    OUTPUT_IMAGE = os.path.join(BASE_DIR, "heatmap_result.jpg")
    
    print("Đang nạp mô hình EfficientNet-B0...")
    try:
        my_model = load_model(MODEL_WEIGHTS, num_classes=10)
        
        print(f"Đang phân tích và sinh bản đồ nhiệt cho ảnh: {TEST_IMAGE}...")
        result_path = generate_gradcam(my_model, TEST_IMAGE, OUTPUT_IMAGE)
        
        print(f"\nThành công! Đã lưu ảnh Grad-CAM tại: {result_path}")
        print("Hãy mở ảnh 'heatmap_result.jpg' lên để xem AI của bạn đang 'nhìn' vào đâu nhé!")
    except FileNotFoundError as e:
        print(f"\nLỗi: {e}")
        print("Hãy chắc chắn bạn đã để 1 bức ảnh tên là 'test_image.jpg' ở thư mục gốc của dự án!")
    except Exception as e:
         print(f"\nCó lỗi hệ thống xảy ra: {e}")