import torch
import torch.nn as nn
from torchvision import models
import os


def export_to_onnx(weights_path: str, output_path: str, num_classes: int = 10):
    # Khởi tạo kiến trúc mô hình
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    # Nạp trọng số đã train
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    
    # Chuyển sang evaluation mode (không dùng dropout, batch norm cố định)
    model.eval()
    
    # Tạo dummy input để model chạy thử
    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=True)
    
    # Xuất file ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )


if __name__ == "__main__":
    # Xác định đường dẫn
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_weights = os.path.join(base_dir, "saved_models", "efficientnet_finetune_best_v2.pth")
    onnx_output = os.path.join(base_dir, "saved_models", "efficientnet_b0.onnx")
    
    print(f"\n{'='*60}")
    print(f"📦 XUẤT MÔ HÌNH ONNX")
    print(f"{'='*60}")
    print(f"PyTorch weights: {model_weights}")
    print(f"ONNX output: {onnx_output}\n")
    
    try:
        # Kiểm tra file weights có tồn tại
        if not os.path.exists(model_weights):
            raise FileNotFoundError(f"Không tìm thấy file: {model_weights}")
        
        pth_size = os.path.getsize(model_weights) / (1024 * 1024)  # Convert to MB
        print(f"Kích thước PyTorch model: {pth_size:.2f} MB")
        
        # Export ONNX
        print("Đang export sang ONNX...")
        export_to_onnx(model_weights, onnx_output)
        
        # Kiểm tra file ONNX đã tạo
        if os.path.exists(onnx_output):
            onnx_size = os.path.getsize(onnx_output) / (1024 * 1024)  # Convert to MB
            compression_ratio = (1 - onnx_size / pth_size) * 100
            
            print(f"\n{'='*60}")
            print(f"✅ XUẤT THÀNH CÔNG!")
            print(f"{'='*60}")
            print(f"Kích thước ONNX model: {onnx_size:.2f} MB")
            print(f"Nén được: {compression_ratio:.1f}% (từ {pth_size:.2f}MB → {onnx_size:.2f}MB)")
            print(f"File lưu tại: {onnx_output}")
            print(f"{'='*60}\n")
        else:
            print("❌ ONNX file không được tạo")
            
    except FileNotFoundError as e:
        print(f"\n❌ LỖI: {e}")
        print("Hãy chạy train_finetune_v2.py để tạo weights trước.")
    except Exception as e:
        print(f"\n❌ LỖI EXPORT: {e}")
        print("Vui lòng kiểm tra lại file weights.")