import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from ultralytics import YOLO

# Настройка CPU/GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. АРХИТЕКТУРА НАШЕЙ СЕТИ НА 1923 КАНАЛА
class SuperTemporalConvNet(nn.Module):
    def __init__(self, in_channels=1923, num_classes=3):
        super(SuperTemporalConvNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.6)
        
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.6)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, num_classes)
        
    def forward(self, x):
        x = self.dropout1(self.relu1(self.bn1(self.conv1(x))))
        x = self.dropout2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

# 2. ЗАГРУЗКА ВСЕХ МОДЕЛЕЙ И ВЕСОВ
WEIGHTS_PATH = "best_super_cnn.pth"
if not os.path.exists(WEIGHTS_PATH):
    print(f"Ошибка: Не найден файл весов {WEIGHTS_PATH} в корне папки.")
    sys.exit(1)

model = SuperTemporalConvNet(in_channels=1923, num_classes=3).to(device)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model.eval()

# Инициализация предобученных экстракторов
mobilenet_global = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT).to(device).eval()
mobilenet_global.classifier = nn.Identity()

mobilenet_local = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT).to(device).eval()
mobilenet_local.classifier = nn.Identity()

yolo_detector = YOLO("yolov8m-pose.pt")

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

CLASS_MAP = {0: "inaction", 1: "move", 2: "work"}

# 3. ЛОГИКА ОБРАБОТКИ ПАПКИ ИЗ 8 КАДРОВ
def predict_folder(folder_path):
    valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    images = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(valid_ext) and not f.startswith('.')])
    
    if len(images) == 0:
        return "Ошибка: В указанной папке нет изображений."
        
    # Адаптация количества кадров строго до 8
    while len(images) < 8: images.append(images[-1])
    images = images[:8]
    
    sequence_features = []
    prev_gray = None
    
    for pth in images:
        frame = cv2.imread(pth)
        if frame is None:
            sequence_features.append(np.zeros(1923))
            continue
            
        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Глобальный эмбеддинг всего кадра
        pil_global = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        tensor_global = preprocess(pil_global).unsqueeze(0).to(device)
        with torch.no_grad():
            feat_global = mobilenet_global(tensor_global).squeeze().cpu().numpy()
            
        # Запуск детектора YOLO
        results = yolo_detector(frame, verbose=False)
        
        # ИСПРАВЛЕНИЕ БАГА ТИПА ДАННЫХ: Вытаскиваем первый результат из списка YOLO результатов
        res = results[0]
        
        bx1, by1, bx2, by2 = 0, 0, 640, 480
        motion_score, flow_x, flow_y = 0.0, 0.0, 0.0
        
        if len(res.boxes) > 0:
            max_area = 0
            for box in res.boxes:
                if int(box.cls[0]) == 0: # Подстраховка для извлечения ID класса
                    # ИСПРАВЛЕНИЕ БАГА РАЗМЕРНОСТИ: Переводим координаты в плоский массив NumPy безопасно
                    coords = box.xyxy.cpu().numpy().astype(int).squeeze()
                    if coords.ndim == 1 and len(coords) == 4:
                        x1, y1, x2, y2 = coords
                        area = (x2 - x1) * (y2 - y1)
                        if area > max_area:
                            max_area = area
                            bx1, by1, bx2, by2 = x1, y1, x2, y2
                            
        bx1, by1 = max(0, bx1), max(0, by1)
        bx2, by2 = min(640, bx2), min(480, by2)
        
        person_crop = frame[by1:by2, bx1:bx2]
        if person_crop.size == 0: 
            person_crop = frame
            
        pil_local = Image.fromarray(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
        tensor_local = preprocess(pil_local).unsqueeze(0).to(device)
        with torch.no_grad():
            feat_local = mobilenet_local(tensor_local).squeeze().cpu().numpy()
            
        # Расчет локального оптического потока
        if prev_gray is not None:
            crop_curr = gray[by1:by2, bx1:bx2]
            crop_prev = prev_gray[by1:by2, bx1:bx2]
            if crop_curr.size > 0 and crop_curr.shape == crop_prev.shape:
                motion_score = np.mean(cv2.absdiff(crop_curr, crop_prev) > 20)
                flow = cv2.calcOpticalFlowFarneback(crop_prev, crop_curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                flow_x, flow_y = np.mean(flow[..., 0]), np.mean(flow[..., 1])
                
        prev_gray = gray
        combined_feat = np.concatenate([feat_global, feat_local, [motion_score, flow_x, flow_y]])
        sequence_features.append(combined_feat)
        
    # Тензор для 1D-CNN: (Batch=1, Channels=1923, Time=8)
    feat_tensor = torch.tensor(np.array(sequence_features), dtype=torch.float32).transpose(0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(feat_tensor)
        predicted_idx = torch.argmax(outputs, dim=1).item()
        
    return CLASS_MAP[predicted_idx]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python predict.py /путь/к/папке/с/8_картинками")
        sys.exit(1)
        
    target_folder = sys.argv[1]
    if not os.path.exists(target_folder):
        print(f"Ошибка: Путь {target_folder} не существует.")
        sys.exit(1)
        
    result_class = predict_folder(target_folder)
    print(f"\n[РЕЗУЛЬТАТ]: {result_class}")
