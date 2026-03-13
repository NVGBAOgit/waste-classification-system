import torch
from torchvision import datasets, transforms #datasets để đọc ảnh, transforms để xử lý ảnh
from torch.utils.data import DataLoader, Subset #dataloader sẽ chia dữ liệu thành các phần nhỏ để gửi vào model xử lý, Subset sẽ tạo tập con từ dataset theo danh sách index
from sklearn.model_selection import train_test_split #chia dữ liệu theo tỷ lệ, đảm bảo các nhãn được chia đều vào 3 tập

# Đường dẫn dataset
DATASET_PATH = "data"

# Tiền xử lý ảnh cho tập train (có augmentation: mỗi ảnh đưa vào 1 lần, mỗi lần có thể là nguyên mẫu hoặc đã qua biến đổi)
train_transforms = transforms.Compose([   # transforms.Compose định dạng lại dữ liệu (ảnh)
    transforms.Resize((224, 224)),        # resize về 224x224
    transforms.RandomHorizontalFlip(),    # lật ảnh ngẫu nhiên
    transforms.RandomRotation(10),        # xoay ảnh ngẫu nhiên 10 độ
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # thay đổi độ sáng
    transforms.ToTensor(),                # chuyển ảnh thành tensor (ma trận số (R, G, B))
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # chuẩn hóa theo chỉ số đặc trưng của ImageNet
])

# Tiền xử lý ảnh cho tập val và test (không augmentation)
val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load dataset 2 lần: train dùng train_transforms, val/test dùng val_test_transforms, dán nhãn cho ảnh
train_dataset = datasets.ImageFolder(DATASET_PATH, transform=train_transforms)
val_test_dataset = datasets.ImageFolder(DATASET_PATH, transform=val_test_transforms)

# Lấy danh sách index và nhãn
total = len(train_dataset) # số ảnh trong dataset: 2527
indices = list(range(total))
labels = [train_dataset.targets[i] for i in indices] # lấy nhãn từng ảnh theo index

# Chia train/val/test theo tỉ lệ 70/15/15, đảm bảo các nhãn chia đều (stratify)
train_indices, val_test_indices = train_test_split(
    indices, test_size=0.3, stratify=labels, random_state=42
) # giá trị 2 biến là tập các index đã được xáo trộn ngẫu nhiên theo tỷ lệ 7 train và 3 test. Cố định cách xáo để so sánh được 3 model sau này
val_indices, test_indices = train_test_split(
    val_test_indices, test_size=0.5, random_state=42
)

# Tạo Subset từ index đã chia
train_set = Subset(train_dataset, train_indices)      
val_set = Subset(val_test_dataset, val_indices)       
test_set = Subset(val_test_dataset, test_indices)     

# Tạo DataLoader
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)   # shuffle để xáo trộn thứ tự mỗi epoch
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)      # không cần shuffle vì chỉ để đánh giá
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)    # không cần shuffle vì chỉ để đánh giá
