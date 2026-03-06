# train_vision.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import copy
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Import hàm load dữ liệu từ file cũ
from prepare_data import load_and_process_data

# --- 1. CẤU HÌNH (CONFIG) ---
BATCH_SIZE = 16       # Số lượng ảnh học một lần (tăng lên 32 nếu GPU mạnh)
NUM_EPOCHS = 10       # Số lần học lặp lại toàn bộ dữ liệu (càng lâu càng kỹ)
LEARNING_RATE = 0.001 # Tốc độ học (0.001 là chuẩn cho fine-tuning)
IMG_SIZE = 224        # Kích thước ảnh đầu vào
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"⚙️ Thiết bị training: {DEVICE}")

# --- 2. CHUẨN BỊ DATASET ---
class IUXrayDataset(Dataset):
    def __init__(self, data_list, transform=None, label_encoder=None):
        self.data = data_list
        self.transform = transform
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = item['image_path']
        label_str = item['true_label']
        
        # Load ảnh
        image = Image.open(img_path).convert("RGB")
        
        # Biến đổi ảnh (Augmentation)
        if self.transform:
            image = self.transform(image)
        
        # Chuyển nhãn chữ sang số
        label = self.label_encoder.transform([label_str])[0]
        
        return image, torch.tensor(label, dtype=torch.long)

# --- 3. HÀM CHÍNH ---
def main():
    # A. Load dữ liệu thô
    print("📥 Đang tải dữ liệu thô...")
    full_data = load_and_process_data() # Lấy từ prepare_data.py
    
    if not full_data:
        print("❌ Không có dữ liệu để train!")
        return

    # B. Chuẩn bị nhãn (Labels)
    # Lấy danh sách tất cả các nhãn để mã hóa
    all_labels = [item['true_label'] for item in full_data]
    le = LabelEncoder()
    le.fit(all_labels)
    num_classes = len(le.classes_)
    
    print(f"🏷️ Tìm thấy {num_classes} loại bệnh: {le.classes_}")
    
    # C. Chia Train/Val (80% Train, 20% Val)
    train_data, val_data = train_test_split(full_data, test_size=0.2, random_state=42, stratify=all_labels)
    print(f"📊 Dữ liệu: Train={len(train_data)} | Val={len(val_data)}")

    # D. Định nghĩa Transforms (Tăng cường ảnh)
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(), # Lật ngang ngẫu nhiên
        transforms.RandomRotation(10),     # Xoay nhẹ +/- 10 độ
        transforms.ColorJitter(brightness=0.1, contrast=0.1), # Chỉnh sáng tối nhẹ
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # E. Tạo DataLoader
    train_dataset = IUXrayDataset(train_data, transform=train_transforms, label_encoder=le)
    val_dataset = IUXrayDataset(val_data, transform=val_transforms, label_encoder=le)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- 4. XÂY DỰNG MÔ HÌNH ---
    print("🏗️ Đang khởi tạo DenseNet121...")
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    
    # Thay thế lớp cuối cùng (Classifier) để phù hợp với số bệnh của chúng ta
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
    
    model = model.to(DEVICE)

    # Hàm loss và Optimizer
    criterion = nn.CrossEntropyLoss()
    # Chỉ train lớp classifier nhanh hơn, hoặc train toàn bộ thì chậm hơn nhưng tốt hơn. 
    # Ở đây ta train toàn bộ nhưng learning rate thấp.
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Scheduler để giảm LR nếu không cải thiện
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # --- 5. VÒNG LẶP TRAIN (TRAINING LOOP) ---
    print(f"🚀 Bắt đầu Train trong {NUM_EPOCHS} epochs...")
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(NUM_EPOCHS):
        print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}')
        print('-' * 10)

        # Mỗi epoch có 2 pha: Train và Evaluate
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() # Chế độ học
                dataloader = train_loader
            else:
                model.eval()  # Chế độ thi
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            # Lặp qua từng batch
            for inputs, labels in dataloader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                # Reset gradients
                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward (chỉ khi train)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Thống kê
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Lưu model tốt nhất
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # Lưu file ngay lập tức
                torch.save(model.state_dict(), 'best_vision_model.pth')
                # Lưu class names để dùng lại sau này
                with open("classes.txt", "w") as f:
                    f.write("\n".join(le.classes_))
                print(f"💾 Đã lưu model tốt nhất mới! (Acc: {best_acc:.4f})")

    print(f'\n🏁 Hoàn tất Training. Accuracy tốt nhất trên tập Val: {best_acc:.4f}')

if __name__ == "__main__":
    main()