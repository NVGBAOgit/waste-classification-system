from fastapi import FastAPI, UploadFile, File
from contextlib import asynccontextmanager
import uvicorn
import numpy as np
from PIL import Image
import io
import os
import cv2
import base64
import uuid

import onnxruntime as ort
from gradcam_helper import load_model as load_gradcam_model, generate_gradcam

# CẤU HÌNH
base_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(base_dir, '..'))

ONNX_MODEL_PATH = os.path.join(project_dir, 'fine_tune_efficientnet', 'saved_models', 'efficientnet_b0.onnx')
PYTORCH_MODEL_PATH = os.path.join(project_dir, 'fine_tune_efficientnet', 'saved_models', 'efficientnet_finetune_best.pth')

CLASS_NAMES = ['battery', 'cardboard', 'clothes', 'glass', 'metal', 
               'organic', 'paper', 'plastic', 'shoes', 'trash']

# Biến toàn cục
ort_session = None
gradcam_model = None


def preprocess_image_onnx(image: Image.Image) -> np.ndarray:
    img = image.resize((224, 224))
    img_data = np.array(img).astype(np.float32) / 255.0
    
    # Chuẩn hóa ImageNet (float32 để tránh auto-cast lên float64)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_data = (img_data - mean) / std
    
    # Đổi trục (H, W, C) → (C, H, W)
    img_data = np.transpose(img_data, (2, 0, 1))
    
    # Thêm batch dimension
    img_data = np.expand_dims(img_data, axis=0)
    
    return img_data.astype(np.float32)


def get_gradcam_focus(heatmap_data: np.ndarray) -> str:
    # Tìm điểm có giá trị lớn nhất (vùng đỏ nhất)
    y, x = np.unravel_index(np.argmax(heatmap_data), heatmap_data.shape)
    h, w = heatmap_data.shape
    
    # Phân vùng theo 3x3 grid
    if y < h/3: pos_y = "phía trên"
    elif y > 2*h/3: pos_y = "phía dưới"
    else: pos_y = "trung tâm"
        
    if x < w/3: pos_x = "bên trái"
    elif x > 2*w/3: pos_x = "bên phải"
    else: pos_x = "ở giữa"
    
    return f"{pos_y} {pos_x} của vật thể"


# LIFESPAN - KHỞI ĐỘNG VÀ TẮT SERVER
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Quản lý vòng đời ứng dụng - load model khi khởi động, giải phóng khi tắt."""
    global ort_session, gradcam_model
    
    print(f"\n{'='*60}")
    print("🚀 KHỞI ĐỘNG HỆ THỐNG")
    print(f"{'='*60}")
    
    try:
        # Load ONNX Runtime (suy luận siêu nhanh)
        if os.path.exists(ONNX_MODEL_PATH):
            ort_session = ort.InferenceSession(ONNX_MODEL_PATH)
            print("✅ ONNX Runtime loaded")
        else:
            print(f"❌ Không tìm ONNX model: {ONNX_MODEL_PATH}")

        # Load PyTorch model (cho Grad-CAM)
        if os.path.exists(PYTORCH_MODEL_PATH):
            gradcam_model = load_gradcam_model(PYTORCH_MODEL_PATH, num_classes=10)
            print("✅ PyTorch model loaded (Grad-CAM)")
        else:
            print(f"❌ Không tìm PyTorch model: {PYTORCH_MODEL_PATH}")
            
    except Exception as e:
        print(f"❌ Lỗi khởi động: {e}")
        
    yield
    print("\n👋 Server đang tắt...")


app = FastAPI(title="GreenAI Advanced Backend", lifespan=lifespan)


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # Kiểm tra model đã load chưa
    if ort_session is None:
        return {"error": "ONNX model not loaded"}
    
    # Đọc ảnh
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
    except Exception as e:
        return {"error": f"Invalid image: {e}"}
    
    # ONNX PREDICTION
    img_array = preprocess_image_onnx(image)
    ort_inputs = {ort_session.get_inputs()[0].name: img_array}
    ort_outs = ort_session.run(None, ort_inputs)
    
    # Softmax
    logits = ort_outs[0][0]
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()
    
    # Top 3 predictions
    top_idxs = np.argsort(probs)[-3:][::-1]
    best_class = CLASS_NAMES[top_idxs[0]]
    best_conf = float(probs[top_idxs[0]] * 100)
    other_preds = [
        {"name": CLASS_NAMES[top_idxs[1]], "conf": float(probs[top_idxs[1]] * 100)},
        {"name": CLASS_NAMES[top_idxs[2]], "conf": float(probs[top_idxs[2]] * 100)}
    ]
    
    # GRAD-CAM EXPLANATION
    heatmap_base64 = ""
    gradcam_focus = "toàn bộ vật thể"
    
    if gradcam_model is not None:
        temp_id = uuid.uuid4().hex
        temp_img_path = os.path.join(base_dir, f"temp_{temp_id}_upload.jpg")
        temp_heatmap_path = os.path.join(base_dir, f"temp_{temp_id}_heatmap.jpg")
        
        try:
            image.save(temp_img_path)
            generate_gradcam(gradcam_model, temp_img_path, temp_heatmap_path)
            
            # Phân tích heatmap để lấy vùng trọng tâm
            heat_map_array = cv2.imread(temp_heatmap_path, cv2.IMREAD_GRAYSCALE)
            if heat_map_array is not None:
                gradcam_focus = get_gradcam_focus(heat_map_array)
            
            # Encode heatmap thành Base64
            with open(temp_heatmap_path, "rb") as img_file:
                heatmap_base64 = base64.b64encode(img_file.read()).decode("utf-8")
                
        except Exception as e:
            print(f"Grad-CAM error: {e}")
            # Tiếp tục trả về kết quả dù Grad-CAM lỗi
            
        finally:
            # ✅ LUÔN DỌN DẸP FILE TẠM dù có crash hay không
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)
            if os.path.exists(temp_heatmap_path):
                os.remove(temp_heatmap_path)

    # Trả về kết quả
    return {
        "best_class": best_class,
        "best_conf": best_conf,
        "other_preds": other_preds,
        "heatmap_base64": heatmap_base64,
        "gradcam_focus": gradcam_focus
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)