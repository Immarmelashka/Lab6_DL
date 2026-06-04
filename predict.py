import os
import sys
import cv2
import glob
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models.video import r3d_18
from PIL import Image
from ultralytics import YOLO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- АРХИТЕКТУРА МОДЕЛИ 1: 1D-CNN (1923 КАНАЛА) ---
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

# --- ИНИЦИАЛИЗАЦИЯ И ЗАГРУЗКА АНСАМБЛЯ ---
cnn_model = SuperTemporalConvNet(in_channels=1923, num_classes=3).to(device)
if os.path.exists("best_super_cnn.pth"):
    cnn_model.load_state_dict(torch.load("best_super_cnn.pth", map_location=device))
cnn_model.eval()

# Инициализация 3D-ResNet18 в соответствии со структурой обучения (с Dropout)
resnet3d_model = r3d_18()
in_features = resnet3d_model.fc.in_features
resnet3d_model.fc = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(in_features, 3)
)
if os.path.exists("best_3d_resnet.pth"):
    resnet3d_model.load_state_dict(torch.load("best_3d_resnet.pth", map_location=device))
resnet3d_model.eval()

# Загрузка экстракторов фич
mobilenet_global = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT).to(device).eval()
mobilenet_global.classifier = nn.Identity()

mobilenet_local = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT).to(device).eval()
mobilenet_local.classifier = nn.Identity()

yolo_detector = YOLO("yolov8m-pose.pt")

preprocess_cnn = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

CLASS_MAP = {0: "inaction", 1: "move", 2: "work"}

def predict_ensemble_track(folder_path):
    valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    all_images = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(valid_ext) and not f.startswith('.')])
    
    if len(all_images) == 0:
        return "Ошибка: В указанной папке нет подходящих изображений."

    # --- Поток 1: 1D-CNN (8 равномерных кадров из трека) ---
    indices = np.linspace(0, len(all_images) - 1, 8, dtype=int)
    cnn_images = [all_images[i] for i in indices]
    
    sequence_features = []
    prev_gray = None
    
    for pth in cnn_images:
        frame = cv2.imread(pth)
        if frame is None:
            sequence_features.append(np.zeros(1923))
            continue
        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        pil_global = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        feat_global = mobilenet_global(preprocess_cnn(pil_global).unsqueeze(0).to(device)).squeeze().detach().cpu().numpy()
        
        results = yolo_detector(frame, verbose=False)
        res = results[0]
        
        bx1, by1, bx2, by2 = 0, 0, 640, 480
        motion_score, flow_x, flow_y = 0.0, 0.0, 0.0
        
        if len(res.boxes) > 0:
            max_area = 0
            for box in res.boxes:
                if int(box.cls) == 0: # Только класс Person
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
        feat_local = mobilenet_local(preprocess_cnn(pil_local).unsqueeze(0).to(device)).squeeze().detach().cpu().numpy()
        
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
        
    feat_tensor = torch.tensor(np.array(sequence_features), dtype=torch.float32).transpose(0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        probs_cnn = torch.softmax(cnn_model(feat_tensor), dim=1).squeeze().cpu().numpy()

    # --- Поток 2: 3D-ResNet18 (Скользящее окно по 16 кадров) ---
    window_probs = []
    window_indices = np.linspace(0, max(1, len(all_images) - 16), min(5, len(all_images)//16 + 1), dtype=int)
    
    for start_f in window_indices:
        clip_paths = all_images[start_f : start_f + 16]
        if len(clip_paths) < 16: 
            break
        
        frames_3d = []
        for pth in clip_paths:
            img = cv2.imread(pth)
            if img is None: 
                continue
            img = cv2.resize(img, (112, 112))
            frames_3d.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
        if len(frames_3d) == 16:
            video = np.array(frames_3d, dtype=np.float32) / 255.0
            video = np.transpose(video, (3, 0, 1, 2))
            video = (video - np.array([0.485, 0.456, 0.406])[:, None, None, None]) / np.array([0.229, 0.224, 0.225])[:, None, None, None]
            video_tensor = torch.tensor(video, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                probs_res = torch.softmax(resnet3d_model(video_tensor), dim=1).squeeze().cpu().numpy()
                window_probs.append(probs_res)
                
    probs_resnet = np.mean(window_probs, axis=0) if len(window_probs) > 0 else np.array([0.33, 0.33, 0.33])

    # --- Мягкое голосование (Soft Voting) ---
    w_resnet = np.array([0.65, 0.30, 0.30])
    w_cnn = np.array([0.35, 0.70, 0.70])
    
    final_probs = (probs_resnet * w_resnet) + (probs_cnn * w_cnn)
    return CLASS_MAP[np.argmax(final_probs)]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python predict.py /путь/к/папке_с_изображениями")
        sys.exit(1)
        
    # ИСПРАВЛЕНИЕ: берем строго первый аргумент пути [1]
    target_folder = sys.argv[1]
    if not os.path.exists(target_folder):
        print(f"Ошибка: Путь {target_folder} не существует.")
        sys.exit(1)
        
    print(f"\n[АНСАМБЛЕВЫЙ РЕЗУЛЬТАТ ДЛЯ ТРЕКА]: {predict_ensemble_track(target_folder)}")
