import json

cells = []

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source if isinstance(source, list) else [source]}

def code(source_lines):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source_lines if isinstance(source_lines, list) else [source_lines]}

def src(*lines):
    """Helper: join lines with newlines for code cell source."""
    result = []
    for line in lines:
        result.append(line + '\n')
    if result:
        result[-1] = result[-1].rstrip('\n')
    return result

# ── Cell 0: Fix sys.path agar library dari codegeex-agent terbaca ─────────────
cells.append(md("# DAC Find IT! 2026 – Face Anti-Spoofing Detection\n"
                "### Notebook Lokal (Windows, RTX 2060)\n"
                "Pastikan dataset sudah didownload dan diekstrak ke folder `./dataset/`"))

cells.append(code(src(
    "# FIX: Pastikan Jupyter membaca library yang sudah diinstall di codegeex-agent env",
    "import sys, os",
    "for _p in [",
    r"    r'c:\users\admin\.codegeex\mamba\envs\codegeex-agent\Lib\site-packages',",
    r"    r'c:\users\admin\.codegeex\mamba\envs\codegeex-agent\lib\site-packages',",
    "]:",
    "    if os.path.exists(_p) and _p not in sys.path:",
    "        sys.path.insert(0, _p)",
    "print('sys.path updated!')"
)))

# ── Cell 1: Imports ──────────────────────────────────────────────────────────
cells.append(md("## 1. Import Library"))
cells.append(code(src(
    "import torch",
    "import torch.nn as nn",
    "from torch.utils.data import Dataset, DataLoader",
    "from torchvision import transforms",
    "import timm",
    "from PIL import Image",
    "from sklearn.model_selection import StratifiedKFold",
    "from sklearn.metrics import f1_score, confusion_matrix, classification_report",
    "import pandas as pd",
    "import numpy as np",
    "import matplotlib.pyplot as plt",
    "import seaborn as sns",
    "import os, random, warnings",
    "from collections import Counter",
    "",
    "warnings.filterwarnings('ignore')",
    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
    "print(f'Using device: {device}')",
    "if torch.cuda.is_available():",
    "    print(f'GPU: {torch.cuda.get_device_name(0)}')",
    "    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')"
)))

# ── Cell 2: Download Dataset dari Kaggle ─────────────────────────────────────
cells.append(md("## 2. Download Dataset dari Kaggle\nJalankan cell ini **sekali saja**. Skip jika dataset sudah ada."))
cells.append(code(src(
    "import os",
    "",
    "# Set Kaggle API token langsung via environment variable",
    "os.environ['KAGGLE_USERNAME'] = 'your_kaggle_username'   # <-- ganti dengan username Kaggle kamu",
    "os.environ['KAGGLE_KEY'] = 'KGAT_515cf194ee486da0c9debd57815d0f28'",
    "",
    "dataset_path = './dataset/train'",
    "if os.path.exists(dataset_path):",
    "    print('Dataset sudah ada, skip download.')",
    "else:",
    "    print('Downloading dataset...')",
    "    os.system('kaggle competitions download -c data-analytics-competition-dac-find-it-2026')",
    "    os.system('tar -xf data-analytics-competition-dac-find-it-2026.zip -C ./ || python -c \"import zipfile; zipfile.ZipFile(\\\"data-analytics-competition-dac-find-it-2026.zip\\\").extractall(\\'./dataset\\')\"')",
    "    print('Selesai!')"
)))

# ── Cell 3: Eksplorasi Dataset ───────────────────────────────────────────────
cells.append(md("## 3. Eksplorasi Dataset (EDA)"))
cells.append(code(src(
    "train_path = './dataset/train'",
    "",
    "print('=== JUMLAH GAMBAR PER KELAS ===')",
    "class_names = sorted(os.listdir(train_path))",
    "total = 0",
    "for folder in class_names:",
    "    count = len(os.listdir(os.path.join(train_path, folder)))",
    "    total += count",
    "    print(f'  {folder:20s}: {count:5d} gambar')",
    "print(f'  {\"TOTAL\":20s}: {total:5d} gambar')",
    "",
    "print(f'\\nTotal test: {len(os.listdir(\"./dataset/test\"))}')"
)))

# ── Cell 4: Visualisasi Sampel ───────────────────────────────────────────────
cells.append(md("### Visualisasi Sampel Tiap Kelas"))
cells.append(code(src(
    "train_path = './dataset/train'",
    "classes = sorted(os.listdir(train_path))",
    "",
    "fig, axes = plt.subplots(2, 3, figsize=(15, 10))",
    "axes = axes.flatten()",
    "",
    "for i, cls in enumerate(classes):",
    "    folder = os.path.join(train_path, cls)",
    "    img_name = random.choice(os.listdir(folder))",
    "    img = Image.open(os.path.join(folder, img_name)).convert('RGB')",
    "    axes[i].imshow(img)",
    "    axes[i].set_title(f'{cls}\\n({img.size[0]}x{img.size[1]})', fontsize=11)",
    "    axes[i].axis('off')",
    "",
    "plt.suptitle('Sample Gambar Tiap Kelas', fontsize=16, fontweight='bold')",
    "plt.tight_layout()",
    "plt.show()"
)))

# ── Cell 5: Prepare DataFrame ────────────────────────────────────────────────
cells.append(md("## 4. Siapkan DataFrame Training"))
cells.append(code(src(
    "train_path = './dataset/train'",
    "data = []",
    "for label in os.listdir(train_path):",
    "    folder = os.path.join(train_path, label)",
    "    if os.path.isdir(folder):",
    "        for img_name in os.listdir(folder):",
    "            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):",
    "                data.append({'path': os.path.join(folder, img_name), 'label': label})",
    "",
    "df = pd.DataFrame(data)",
    "label2idx = {l: i for i, l in enumerate(sorted(df['label'].unique()))}",
    "idx2label = {i: l for l, i in label2idx.items()}",
    "df['label_idx'] = df['label'].map(label2idx)",
    "",
    "print('Distribusi kelas:')",
    "print(df['label'].value_counts())",
    "print('\\nLabel mapping:')",
    "for k, v in label2idx.items():",
    "    print(f'  {k}: {v}')"
)))

# ── Cell 6: Dataset & Augmentasi ─────────────────────────────────────────────
cells.append(md("## 5. Dataset Class & Augmentasi\n> Menggunakan PadToSquare agar aspect ratio wajah tidak distorsi (temuan EDA 1)"))
cells.append(code(src(
    "import torchvision.transforms.functional as TF",
    "",
    "class PadToSquare:",
    "    \"\"\"Tambah border hitam agar gambar jadi persegi sebelum di-resize.\"\"\"",
    "    def __call__(self, img):",
    "        w, h = img.size",
    "        max_wh = max(w, h)",
    "        p_left   = (max_wh - w) // 2",
    "        p_right  = max_wh - w - p_left",
    "        p_top    = (max_wh - h) // 2",
    "        p_bottom = max_wh - h - p_top",
    "        return TF.pad(img, (p_left, p_top, p_right, p_bottom), 0, 'constant')",
    "",
    "class FaceDataset(Dataset):",
    "    def __init__(self, df, transform=None):",
    "        self.df = df.reset_index(drop=True)",
    "        self.transform = transform",
    "    def __len__(self):",
    "        return len(self.df)",
    "    def __getitem__(self, idx):",
    "        row = self.df.iloc[idx]",
    "        img = Image.open(row['path']).convert('RGB')",
    "        if self.transform:",
    "            img = self.transform(img)",
    "        return img, row['label_idx']",
    "",
    "IMG_SIZE = 224",
    "",
    "train_transform = transforms.Compose([",
    "    PadToSquare(),",
    "    transforms.Resize((IMG_SIZE, IMG_SIZE)),",
    "    transforms.RandomHorizontalFlip(),",
    "    transforms.RandomRotation(15),",
    "    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),",
    "    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),",
    "    transforms.ToTensor(),",
    "    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),",
    "    transforms.RandomErasing(p=0.3),",
    "])",
    "",
    "val_transform = transforms.Compose([",
    "    PadToSquare(),",
    "    transforms.Resize((IMG_SIZE, IMG_SIZE)),",
    "    transforms.ToTensor(),",
    "    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),",
    "])",
    "",
    "print('Transform pipeline siap!')"
)))

# ── Cell 7: Training Function ─────────────────────────────────────────────────
cells.append(md("## 6. Training Function (5-Fold)"))
cells.append(code(src(
    "def train_one_fold(df, train_idx, val_idx, fold_num):",
    "    print(f'\\n===== FOLD {fold_num} =====')",
    "    train_df = df.iloc[train_idx]",
    "    val_df   = df.iloc[val_idx]",
    "",
    "    train_dataset = FaceDataset(train_df, train_transform)",
    "    val_dataset   = FaceDataset(val_df,   val_transform)",
    "",
    "    # Weighted sampler untuk handle class imbalance",
    "    class_counts = Counter(train_df['label_idx'].values)",
    "    weights  = [1.0 / class_counts[l] for l in train_df['label_idx'].values]",
    "    sampler  = torch.utils.data.WeightedRandomSampler(weights, len(weights))",
    "",
    "    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=2, pin_memory=True)",
    "    val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False,   num_workers=2, pin_memory=True)",
    "",
    "    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=6)",
    "    model = model.to(device)",
    "",
    "    total = sum(class_counts.values())",
    "    class_weights = torch.tensor([total / (6 * class_counts[i]) for i in range(6)], dtype=torch.float).to(device)",
    "    criterion = nn.CrossEntropyLoss(weight=class_weights)",
    "    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)",
    "    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)",
    "",
    "    best_f1, best_state = 0, None",
    "",
    "    for epoch in range(15):",
    "        model.train()",
    "        train_loss = 0",
    "        for imgs, labels in train_loader:",
    "            imgs, labels = imgs.to(device), labels.to(device)",
    "            optimizer.zero_grad()",
    "            loss = criterion(model(imgs), labels)",
    "            loss.backward()",
    "            optimizer.step()",
    "            train_loss += loss.item()",
    "        scheduler.step()",
    "",
    "        model.eval()",
    "        all_preds, all_labels = [], []",
    "        with torch.no_grad():",
    "            for imgs, labels in val_loader:",
    "                preds = model(imgs.to(device)).argmax(dim=1).cpu().numpy()",
    "                all_preds.extend(preds)",
    "                all_labels.extend(labels.numpy())",
    "",
    "        macro_f1 = f1_score(all_labels, all_preds, average='macro')",
    "        print(f'  Epoch {epoch+1:2d} | Loss: {train_loss/len(train_loader):.4f} | Macro F1: {macro_f1:.4f}')",
    "",
    "        if macro_f1 > best_f1:",
    "            best_f1 = macro_f1",
    "            best_state = model.state_dict().copy()",
    "",
    "    print(f'  >> Best Macro F1 Fold {fold_num}: {best_f1:.4f}')",
    "    model.load_state_dict(best_state)",
    "    return model, best_f1"
)))

# ── Cell 8: Jalankan Training ─────────────────────────────────────────────────
cells.append(md("## 7. Jalankan 5-Fold Training"))
cells.append(code(src(
    "skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
    "models, scores = [], []",
    "",
    "for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label_idx'])):",
    "    m, s = train_one_fold(df, train_idx, val_idx, fold + 1)",
    "    models.append(m)",
    "    scores.append(s)",
    "",
    "print(f'\\n===== HASIL AKHIR =====')",
    "print(f'Average Macro F1 : {np.mean(scores):.4f}')",
    "print(f'Std Dev          : {np.std(scores):.4f}')",
    "for i, s in enumerate(scores):",
    "    print(f'  Fold {i+1}: {s:.4f}')"
)))

# ── Cell 9: Confusion Matrix Ensemble ─────────────────────────────────────────
cells.append(md("## 8. Confusion Matrix (Ensemble Prediksi)"))
cells.append(code(src(
    "folds    = list(skf.split(df, df['label_idx']))",
    "_, val_idx = folds[-1]",
    "val_df   = df.iloc[val_idx]",
    "val_dataset = FaceDataset(val_df, val_transform)",
    "val_loader  = DataLoader(val_dataset, batch_size=32, shuffle=False)",
    "",
    "all_probs = []",
    "for m in models:",
    "    m.eval()",
    "    fold_probs = []",
    "    with torch.no_grad():",
    "        for imgs, _ in val_loader:",
    "            probs = torch.softmax(m(imgs.to(device)), dim=1).cpu().numpy()",
    "            fold_probs.append(probs)",
    "    all_probs.append(np.concatenate(fold_probs))",
    "",
    "avg_probs   = np.mean(all_probs, axis=0)",
    "preds       = np.argmax(avg_probs, axis=1)",
    "true_labels = val_df['label_idx'].values",
    "label_names = [idx2label[i] for i in range(6)]",
    "",
    "print(classification_report(true_labels, preds, target_names=label_names))",
    "",
    "cm = confusion_matrix(true_labels, preds)",
    "fig, ax = plt.subplots(figsize=(10, 8))",
    "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',",
    "            xticklabels=label_names, yticklabels=label_names)",
    "plt.title('Confusion Matrix – Ensemble (Fold Terakhir)', fontsize=14, fontweight='bold')",
    "plt.xlabel('Predicted')",
    "plt.ylabel('Actual')",
    "plt.tight_layout()",
    "plt.savefig('confusion_matrix_ensemble.png', dpi=150)",
    "plt.show()",
    "print('Confusion matrix disimpan sebagai confusion_matrix_ensemble.png')"
)))

# ── Cell 10: Inferensi & Submission ───────────────────────────────────────────
cells.append(md("## 9. Inferensi Test Set & Buat submission.csv"))
cells.append(code(src(
    "test_path = './dataset/test'",
    "test_files = sorted([f for f in os.listdir(test_path) if f.lower().endswith(('.jpg','.jpeg','.png'))])",
    "",
    "test_data = [{'path': os.path.join(test_path, f), 'id': os.path.splitext(f)[0]} for f in test_files]",
    "test_df   = pd.DataFrame(test_data)",
    "",
    "class TestDataset(Dataset):",
    "    def __init__(self, df, transform):",
    "        self.df = df.reset_index(drop=True)",
    "        self.transform = transform",
    "    def __len__(self): return len(self.df)",
    "    def __getitem__(self, idx):",
    "        img = Image.open(self.df.iloc[idx]['path']).convert('RGB')",
    "        return self.transform(img)",
    "",
    "test_dataset = TestDataset(test_df, val_transform)",
    "test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)",
    "",
    "# Ensemble prediksi dari semua fold",
    "all_test_probs = []",
    "for m in models:",
    "    m.eval()",
    "    fold_probs = []",
    "    with torch.no_grad():",
    "        for imgs in test_loader:",
    "            probs = torch.softmax(m(imgs.to(device)), dim=1).cpu().numpy()",
    "            fold_probs.append(probs)",
    "    all_test_probs.append(np.concatenate(fold_probs))",
    "",
    "avg_test_probs = np.mean(all_test_probs, axis=0)",
    "test_preds     = np.argmax(avg_test_probs, axis=1)",
    "pred_labels    = [idx2label[p] for p in test_preds]",
    "",
    "submission = pd.DataFrame({'id': test_df['id'], 'label': pred_labels})",
    "submission.to_csv('submission.csv', index=False)",
    "print(f'Submission disimpan! Total: {len(submission)} baris')",
    "print(submission['label'].value_counts())",
    "submission.head(10)"
)))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3 (FindIT DAC)", "language": "python", "name": "codegeex-agent"},
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("e:/GIT/My-Playfield/Lomba/Findit/dac_find_it_2026.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print("Notebook berhasil dibuat!")
