import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F
import timm
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using: {device}')

# === Dataset ===
train_path = './data/train'
data = []
if os.path.exists(train_path):
    for label in os.listdir(train_path):
        folder = os.path.join(train_path, label)
        if os.path.isdir(folder):
            for img_name in os.listdir(folder):
                data.append({'path': os.path.join(folder, img_name), 'label': label})

df = pd.DataFrame(data)
if len(df) == 0:
    print("Oops! Dataset kosong atau folder './data/train' tidak memiliki gambar.")
    exit()

label2idx = {l: i for i, l in enumerate(sorted(df['label'].unique()))}
idx2label = {i: l for l, i in label2idx.items()}
df['label_idx'] = df['label'].map(label2idx)
print("Distribusi Kelas Dataset:")
print(df['label'].value_counts())

# Tambahan: Mempertahankan Aspect Ratio dari EDA 1
class PadToSquare(object):
    def __call__(self, img):
        w, h = img.size
        max_wh = max(w, h)
        p_left = (max_wh - w) // 2
        p_right = max_wh - w - p_left
        p_top = (max_wh - h) // 2
        p_bottom = max_wh - h - p_top
        return F.pad(img, (p_left, p_top, p_right, p_bottom), 0, 'constant')

class FaceDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['path']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, row['label_idx']

train_transform = transforms.Compose([
    PadToSquare(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.3),
])

val_transform = transforms.Compose([
    PadToSquare(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# === Training Function ===
def train_one_fold(df, train_idx, val_idx, fold_num):
    print(f'\n===== FOLD {fold_num} =====')
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    train_dataset = FaceDataset(train_df, train_transform)
    val_dataset = FaceDataset(val_df, val_transform)

    class_counts = Counter(train_df['label_idx'].values)
    weights = [1.0 / class_counts[label] for label in train_df['label_idx'].values]
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))

    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=6)
    model = model.to(device)

    total = sum(class_counts.values())
    class_weights = torch.tensor([total / (6 * class_counts[i]) for i in range(6)], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

    best_f1 = 0
    best_model_state = None

    for epoch in range(15): # Epoch statis sesuai template pengguna
        model.train()
        train_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        print(f'Epoch {epoch+1:2d} | Loss: {train_loss/len(train_loader):.4f} | Macro F1: {macro_f1:.4f}')

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_model_state = model.state_dict().copy()

    print(f'Best Fold {fold_num} Macro F1: {best_f1:.4f}')
    model.load_state_dict(best_model_state)
    return model, best_f1

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
models = []
scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label_idx'])):
    model, score = train_one_fold(df, train_idx, val_idx, fold + 1)
    models.append(model)
    scores.append(score)

print(f'\n===== HASIL AKHIR ENSEMBLE =====')
print(f'Average Macro F1: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})')

# Menghitung Confusion Matrix dan menyimpannya ke gambar agar dapat ditinjau
folds = list(skf.split(df, df['label_idx']))
_, val_idx = folds[-1]
val_df = df.iloc[val_idx]
val_dataset = FaceDataset(val_df, val_transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

all_probs = []
for model in models:
    model.eval()
    fold_probs = []
    with torch.no_grad():
        for imgs, _ in val_loader:
            imgs = imgs.to(device)
            outputs = torch.softmax(model(imgs), dim=1)
            fold_probs.append(outputs.cpu().numpy())
    all_probs.append(np.concatenate(fold_probs))

avg_probs = np.mean(all_probs, axis=0)
preds = np.argmax(avg_probs, axis=1)
true_labels = val_df['label_idx'].values

print("\n", classification_report(true_labels, preds, target_names=[idx2label[i] for i in range(6)]))

cm = confusion_matrix(true_labels, preds)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[idx2label[i] for i in range(6)],
            yticklabels=[idx2label[i] for i in range(6)])
plt.title('Confusion Matrix (Ensemble)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix_ensemble.png', dpi=300)
print('Confusion matrix telah disimpan sebagai confusion_matrix_ensemble.png')
