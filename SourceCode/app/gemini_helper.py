import os
import json
import re
from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# BỘ TỪ ĐIỂN DỰ PHÒNG
WASTE_BINS = {
    'battery': 'RÁC NGUY HẠI (Thùng Riêng)',
    'cardboard': 'RÁC TÁI CHẾ (Màu Xanh)',
    'clothes': 'RÁC VÔ CƠ (Màu Cam)',
    'glass': 'RÁC TÁI CHẾ (Màu Xanh)',
    'metal': 'RÁC TÁI CHẾ (Màu Xanh)',
    'organic': 'RÁC HỮU CƠ (Màu Nâu)',
    'paper': 'RÁC TÁI CHẾ (Màu Xanh)',
    'plastic': 'RÁC TÁI CHẾ (Màu Xanh)',
    'shoes': 'RÁC VÔ CƠ (Màu Cam)',
    'trash': 'RÁC VÔ CƠ (Màu Cam)'
}

DEFAULT_ADVICE = {
    'battery': "Bọc kín hai đầu cực bằng băng keo và mang đến điểm thu gom rác nguy hại.",
    'cardboard': "Tháo băng keo, gỡ kim ghim, xếp chồng gọn gàng và bỏ vào thùng rác tái chế.",
    'clothes': "Ủng hộ nếu còn tốt, hoặc cắt nhỏ bỏ vào túi kín.",
    'glass': "Đổ hết chất lỏng, rửa sạch. Dùng giấy báo bọc ngoài nếu vỡ.",
    'metal': "Đổ hết thức ăn, rửa sạch và bóp dẹp lon. Bỏ thùng tái chế.",
    'organic': "Bỏ trực tiếp vào thùng rác hữu cơ, có thể ủ phân compost.",
    'paper': "Giấy sạch gấp gọn bỏ thùng tái chế. Giấy bẩn dính dầu mỡ bỏ rác vô cơ.",
    'plastic': "Đổ cạn nước, bóp dẹp chai nhựa để tiết kiệm không gian.",
    'shoes': "Buộc cặp đôi lại. Tặng nếu còn dùng được, hoặc bỏ thùng rác vô cơ.",
    'trash': "Buộc kín miệng túi rác, không lẫn kim loại hay pin."
}


def get_waste_advice(image, waste_class, confidence, gradcam_focus):
    is_uncertain = confidence < 60.0
    
    # Cố gắng gọi Gemini để phân tích tình trạng chi tiết
    prompt = f"""
    Hệ thống phân loại: '{waste_class}' (Confidence: {confidence:.2f}%).
    Vùng quan sát: {gradcam_focus}
    {"⚠️ Độ tin cậy thấp, hãy đánh giá cẩn thận." if is_uncertain else ""}
    
    NHIỆM VỤ:
    1. Quan sát TÌNH TRẠNG vật thể (sạch/bẩn, khô/ướt, nguyên vẹn/nát...).
    2. Đánh giá khả năng tái chế: 'recyclable' (nếu sạch, có thể tái chế), 'non_recyclable' (nếu bẩn, dính mỡ, không thể tái chế), hoặc 'hazardous' (nếu là pin/đồ điện tử/hóa chất).
    3. TUYỆT ĐỐI KHÔNG gọi tên cụ thể của vật thể.
    
    TRẢ VỀ ĐÚNG ĐỊNH DẠNG JSON NÀY:
    {{
        "recyclability": "recyclable | non_recyclable | hazardous",
        "condition": "[Mô tả tình trạng bằng tiếng Việt]",
        "reason": "[Giải thích ngắn gọn lý do tại sao tái chế được hay không]"
    }}
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=[prompt, image],
            config={"response_mime_type": "application/json"}
        )
        
        # Parse JSON từ Gemini
        match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if not match:
            raise ValueError("Lỗi parse JSON")
            
        gemini_data = json.loads(match.group(0))
        
        rec_status = gemini_data.get("recyclability", "non_recyclable")
        condition = gemini_data.get("condition", "Không xác định")
        reason = gemini_data.get("reason", "Không có lý do")

        # Hệ chuyên gia ra quyết định dựa trên AI
        bin_name = "RÁC VÔ CƠ (Màu Cam)"
        if waste_class in ['plastic', 'glass', 'metal', 'paper', 'cardboard']:
            if rec_status == "recyclable":
                bin_name = "RÁC TÁI CHẾ (Màu Xanh)"
            else:
                bin_name = "RÁC VÔ CƠ (Màu Cam)"
        elif waste_class == 'organic':
            bin_name = "RÁC HỮU CƠ (Màu Nâu)"
        elif waste_class == 'battery':
            bin_name = "RÁC NGUY HẠI (Thùng Riêng)"

        # Trả về kết quả
        return {
            "description": f"Dữ liệu xác định vật liệu thuộc nhóm **{waste_class.upper()}**.\n\n**Phân tích từ AI:** {condition}.",
            "advice": f"**Lý do:** {reason}.\n\n👉 **Quyết định hệ thống:** Bạn hãy bỏ vật thể này vào thùng **{bin_name}**."
        }

    # Phương án dự phòng khi Gemini lỗi
    except Exception as e:
        print(f"Cảnh báo kết nối Cloud LLM: {e} -> Tự động kích hoạt Từ điển Dự phòng.")
        
        fallback_bin = WASTE_BINS.get(waste_class, 'RÁC VÔ CƠ (Màu Cam)')
        fallback_advice = DEFAULT_ADVICE.get(waste_class, "Vui lòng phân loại rác theo quy định địa phương.")
        
        return {
            "description": f"Dữ liệu Offline xác định vật liệu thuộc nhóm **{waste_class.upper()}**.\n\n*(⚠️ Mạng Cloud AI đang gián đoạn. Hệ thống tự động dùng quy tắc tĩnh nội bộ để phản hồi).* ",
            "advice": f"**Hướng dẫn:** {fallback_advice}\n\n👉 **Quyết định hệ thống (Offline):** Vui lòng bỏ vào thùng **{fallback_bin}**."
        }