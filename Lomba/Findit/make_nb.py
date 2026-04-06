import json

def md(source):
    if isinstance(source, str):
        source = [source]
    return {"cell_type": "markdown", "metadata": {}, "source": source}

def code(lines):
    if isinstance(lines, str):
        lines = [lines]
    result = []
    for line in lines:
        result.append(line + '\n')
    if result:
        result[-1] = result[-1].rstrip('\n')
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": result}

cells = []

# ════════════════════════════════════════════════════════════════════════
# HEADER
cells.append(md([
    "# DAC Find IT! 2026 – Face Anti-Spoofing Detection\n",
    "**Notebook Lokal | Windows + RTX 2060**\n\n",
    "Pipeline lengkap: EDA → Preprocessing → Training (5-Fold) → Submission\n",
]))

# ════════════════════════════════════════════════════════════════════════
# FIX KERNEL PATH
cells.append(md("## 0. Setup Environment"))
cells.append(code([
    "# Fix: pastikan library terbaca dari environment yang benar",
    "import sys, os",
    "for _p in [",
    r"    r'c:\users\admin\.codegeex\mamba\envs\codegeex-agent\Lib\site-packages',",
    r"    r'c:\users\admin\.codegeex\mamba\envs\codegeex-agent\lib\site-packages',",
    "]:",
    "    if os.path.exists(_p) and _p not in sys.path:",
    "        sys.path.insert(0, _p)",
    "print('sys.path OK')",
]))

# ════════════════════════════════════════════════════════════════════════
# IMPORTS
cells.append(md("## 1. Import Library"))
cells.append(code([
    "import torch, torch.nn as nn, torch.optim as optim",
    "from torch.utils.data import Dataset, DataLoader",
    "from torchvision import transforms",
    "import torchvision.transforms.functional as TF",
    "import timm",
    "from PIL import Image",
    "from sklearn.model_selection import StratifiedKFold",
    "from sklearn.metrics import f1_score, confusion_matrix, classification_report",
    "import pandas as pd, numpy as np",
    "import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec",
    "import matplotlib.patches as mpatches",
    "import seaborn as sns",
    "import os, random, warnings, hashlib, cv2",
    "from collections import Counter",
    "warnings.filterwarnings('ignore')",
    "",
    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
    "print(f'Device: {device}')",
    "if torch.cuda.is_available():",
    "    print(f'GPU: {torch.cuda.get_device_name(0)}')",
    "    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB')",
]))

# ════════════════════════════════════════════════════════════════════════
# LOAD DATA
cells.append(md("## 2. Load Dataset"))
cells.append(code([
    "train_path = './dataset/train'",
    "test_path  = './dataset/test'",
    "classes    = sorted(os.listdir(train_path))",
    "",
    "data = []",
    "for label in classes:",
    "    folder = os.path.join(train_path, label)",
    "    if not os.path.isdir(folder): continue",
    "    for fname in os.listdir(folder):",
    "        if fname.lower().endswith(('.jpg','.jpeg','.png')):",
    "            data.append({'path': os.path.join(folder, fname), 'label': label})",
    "",
    "df = pd.DataFrame(data)",
    "label2idx = {l: i for i, l in enumerate(sorted(df['label'].unique()))}",
    "idx2label = {i: l for l, i in label2idx.items()}",
    "df['label_idx'] = df['label'].map(label2idx)",
    "",
    "print('=== Distribusi Kelas ===')",
    "print(df['label'].value_counts())",
    "print(f'Total: {len(df)} gambar')",
]))

# ════════════════════════════════════════════════════════════════════════
# EDA 1
cells.append(md([
    "## EDA 1: Statistik Ukuran & Aspect Ratio per Kelas\n",
    "**Tujuan:** Pahami variasi ukuran gambar dan aspect ratio untuk menentukan strategi resize.\n",
]))
cells.append(code([
    "print('=== STATISTIK UKURAN GAMBAR PER KELAS ===')",
    "stats = []",
    "for _, row in df.iterrows():",
    "    try:",
    "        img = Image.open(row['path'])",
    "        w, h = img.size",
    "        fsize = os.path.getsize(row['path']) / 1024",
    "        stats.append({'class': row['label'], 'width': w, 'height': h,",
    "                       'aspect_ratio': round(w/h, 2), 'size_kb': round(fsize, 2)})",
    "    except: pass",
    "",
    "stats_df = pd.DataFrame(stats)",
    "print(stats_df.groupby('class')[['width','height','aspect_ratio','size_kb']].describe().round(1).to_string())",
]))

# ════════════════════════════════════════════════════════════════════════
# EDA 3
cells.append(md([
    "## EDA 3: Visual Inspection per Kelas\n",
    "Lihat sampel gambar + identifikasi visual artifact tiap kelas.\n",
    "Judul merah = filesize < 15KB (low quality).\n",
]))
cells.append(code([
    "N_SAMPLES = 4",
    "fig, axes = plt.subplots(len(classes), N_SAMPLES, figsize=(16, len(classes)*3))",
    "for row, cls in enumerate(classes):",
    "    folder = os.path.join(train_path, cls)",
    "    files  = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg','.jpeg','.png'))]",
    "    samples = random.sample(files, min(N_SAMPLES, len(files)))",
    "    for col, fname in enumerate(samples):",
    "        fpath = os.path.join(folder, fname)",
    "        img   = Image.open(fpath).convert('RGB')",
    "        w, h  = img.size",
    "        fsize = os.path.getsize(fpath)/1024",
    "        color = 'red' if fsize < 15 else 'green'",
    "        axes[row][col].imshow(img)",
    "        axes[row][col].axis('off')",
    "        axes[row][col].set_title(f'{w}x{h} | {fsize:.0f}KB', fontsize=8, color=color)",
    "    axes[row][0].set_ylabel(cls, fontsize=10, fontweight='bold')",
    "plt.suptitle('EDA 3 – Visual Inspection (merah=low quality)', fontsize=13, fontweight='bold', y=1.01)",
    "plt.tight_layout()",
    "plt.savefig('eda3_visual.png', dpi=100, bbox_inches='tight')",
    "plt.show()",
]))

# ════════════════════════════════════════════════════════════════════════
# EDA 4
cells.append(md("## EDA 4: Distribusi File Size & Format"))
cells.append(code([
    "fsize_df = pd.DataFrame(stats)",
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))",
    "",
    "# File size distribution",
    "axes[0].hist(fsize_df['size_kb'], bins=50, color='steelblue', edgecolor='white')",
    "axes[0].set_title('Distribusi Ukuran File', fontweight='bold')",
    "axes[0].set_xlabel('Ukuran (KB)')",
    "axes[0].set_ylabel('Jumlah Gambar')",
    "axes[0].axvline(10, color='red', linestyle='--', label='Threshold 10KB')",
    "axes[0].legend()",
    "",
    "# File size per class boxplot",
    "fsize_data = [fsize_df[fsize_df['class']==c]['size_kb'].values for c in classes]",
    "axes[1].boxplot(fsize_data, labels=[c.replace('fake_','f_') for c in classes], patch_artist=True)",
    "axes[1].set_title('Ukuran File per Kelas', fontweight='bold')",
    "axes[1].set_ylabel('KB')",
    "axes[1].tick_params(axis='x', rotation=25)",
    "",
    "plt.tight_layout()",
    "plt.savefig('eda4_filesize.png', dpi=100)",
    "plt.show()",
    "",
    "low_q = fsize_df[fsize_df['size_kb'] < 10]",
    "print(f'File < 10KB: {len(low_q)}')",
    "if len(low_q): print(low_q[['class','size_kb']].sort_values('size_kb').to_string(index=False))",
]))

# ════════════════════════════════════════════════════════════════════════
# EDA 5
cells.append(md([
    "## EDA 5: Brightness, Warna & Sharpness\n",
    "Menganalisis distribusi kecerahan, channel RGB, dan ketajaman gambar per kelas.\n",
]))
cells.append(code([
    "MAX_PER_CLASS = 100",
    "eda5 = []",
    "for cls in classes:",
    "    folder = os.path.join(train_path, cls)",
    "    files  = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg','.jpeg','.png'))]",
    "    sample = random.sample(files, min(MAX_PER_CLASS, len(files)))",
    "    for fname in sample:",
    "        fpath = os.path.join(folder, fname)",
    "        try:",
    "            img_np = np.array(Image.open(fpath).convert('RGB'))",
    "            gray   = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)",
    "            eda5.append({",
    "                'class': cls,",
    "                'brightness': img_np.mean(),",
    "                'r_mean': img_np[:,:,0].mean(),",
    "                'g_mean': img_np[:,:,1].mean(),",
    "                'b_mean': img_np[:,:,2].mean(),",
    "                'sharpness': cv2.Laplacian(gray, cv2.CV_64F).var()",
    "            })",
    "        except: pass",
    "",
    "e5 = pd.DataFrame(eda5)",
    "print('=== Rata-rata per Kelas ===')",
    "print(e5.groupby('class')[['brightness','sharpness']].mean().round(1).to_string())",
    "",
    "fig, axes = plt.subplots(1, 3, figsize=(18,5))",
    "palette = ['#4CAF50','#FF5722','#2196F3','#9C27B0','#FF9800','#607D8B']",
    "bright_data = [e5[e5['class']==c]['brightness'].values for c in classes]",
    "bp = axes[0].boxplot(bright_data, labels=[c.replace('fake_','f_') for c in classes], patch_artist=True)",
    "[p.set_facecolor(col) for p, col in zip(bp['boxes'], palette)]",
    "axes[0].set_title('Brightness per Kelas', fontweight='bold')",
    "axes[0].tick_params(axis='x', rotation=25)",
    "",
    "x = np.arange(len(classes)); w = 0.25",
    "r_v = [e5[e5['class']==c]['r_mean'].mean() for c in classes]",
    "g_v = [e5[e5['class']==c]['g_mean'].mean() for c in classes]",
    "b_v = [e5[e5['class']==c]['b_mean'].mean() for c in classes]",
    "axes[1].bar(x-w, r_v, w, label='R', color='#E53935', alpha=0.85)",
    "axes[1].bar(x,   g_v, w, label='G', color='#43A047', alpha=0.85)",
    "axes[1].bar(x+w, b_v, w, label='B', color='#1E88E5', alpha=0.85)",
    "axes[1].set_xticks(x)",
    "axes[1].set_xticklabels([c.replace('fake_','f_') for c in classes], rotation=25, ha='right')",
    "axes[1].set_title('Rata-rata Channel RGB', fontweight='bold')",
    "axes[1].legend()",
    "",
    "sharp_data = [e5[e5['class']==c]['sharpness'].values for c in classes]",
    "bp2 = axes[2].boxplot(sharp_data, labels=[c.replace('fake_','f_') for c in classes], patch_artist=True)",
    "[p.set_facecolor(col) for p, col in zip(bp2['boxes'], palette)]",
    "axes[2].set_yscale('log')",
    "axes[2].set_title('Sharpness (log scale)', fontweight='bold')",
    "axes[2].tick_params(axis='x', rotation=25)",
    "",
    "plt.suptitle('EDA 5 – Brightness, Warna & Sharpness', fontsize=13, fontweight='bold')",
    "plt.tight_layout()",
    "plt.savefig('eda5_brightness.png', dpi=100)",
    "plt.show()",
]))

# ════════════════════════════════════════════════════════════════════════
# EDA 6
cells.append(md([
    "## EDA 6: Analisis Tekstur & Frekuensi (FFT)\n",
    "Deteksi pola tersembunyi: Moiré pattern (fake_printed), piksel grid (fake_screen).\n",
]))
cells.append(code([
    "RESIZE = (256, 256)",
    "N_SAMPLE = 40",
    "max_r = RESIZE[0]//2",
    "",
    "fig, ax = plt.subplots(figsize=(12, 5))",
    "palette = ['#4CAF50','#FF5722','#2196F3','#9C27B0','#FF9800','#607D8B']",
    "",
    "for ci, cls in enumerate(classes):",
    "    folder = os.path.join(train_path, cls)",
    "    files  = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg','.jpeg','.png'))]",
    "    sample = random.sample(files, min(N_SAMPLE, len(files)))",
    "    all_power = []",
    "    for fname in sample:",
    "        try:",
    "            img   = Image.open(os.path.join(folder, fname)).convert('L').resize(RESIZE)",
    "            gray  = np.array(img, dtype=np.float32)",
    "            mag   = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray))))",
    "            h, w  = mag.shape; cy, cx = h//2, w//2",
    "            Y, X  = np.ogrid[:h, :w]",
    "            rad   = np.sqrt((X-cx)**2 + (Y-cy)**2).astype(int)",
    "            pwr   = np.array([mag[rad==r].mean() if (rad==r).any() else 0 for r in range(max_r)])",
    "            all_power.append(pwr)",
    "        except: pass",
    "    if all_power:",
    "        ax.plot(np.mean(all_power, axis=0), label=cls, color=palette[ci], linewidth=2)",
    "",
    "ax.set_xlabel('Frekuensi Spatial', fontsize=11)",
    "ax.set_ylabel('Rata-rata Log Power', fontsize=11)",
    "ax.set_title(f'EDA 6 – Rata-rata Radial Power Spectrum ({N_SAMPLE} sampel/kelas)',",
    "             fontsize=13, fontweight='bold')",
    "ax.legend(fontsize=10)",
    "ax.grid(alpha=0.3)",
    "plt.tight_layout()",
    "plt.savefig('eda6_power_spectrum.png', dpi=100)",
    "plt.show()",
]))

# ════════════════════════════════════════════════════════════════════════
# EDA 8 / CLASS IMBALANCE
cells.append(md("## EDA 8: Class Imbalance & Keputusan Final"))
cells.append(code([
    "counts = df['label'].value_counts().sort_index()",
    "total  = counts.sum()",
    "print('=== DISTRIBUSI KELAS ===')",
    "for cls, cnt in counts.items():",
    "    bar = '█' * int(cnt/total*50)",
    "    print(f'  {cls:20s}: {cnt:4d} ({cnt/total*100:5.1f}%) {bar}')",
    "print(f'\\nImbalance ratio: {counts.max()/counts.min():.2f}x')",
    "",
    "fig, axes = plt.subplots(1, 2, figsize=(14,5))",
    "cls_list = sorted(counts.index)",
    "cnt_list = [counts[c] for c in cls_list]",
    "palette  = ['#4CAF50','#FF5722','#2196F3','#9C27B0','#FF9800','#607D8B']",
    "short    = [c.replace('fake_','f_') for c in cls_list]",
    "",
    "bars = axes[0].bar(short, cnt_list, color=palette, edgecolor='white')",
    "[axes[0].text(b.get_x()+b.get_width()/2, b.get_height()+2, str(c),",
    "              ha='center', fontsize=9, fontweight='bold') for b, c in zip(bars, cnt_list)]",
    "axes[0].set_title('Jumlah Gambar per Kelas', fontweight='bold')",
    "axes[0].axhline(total/len(cls_list), color='red', linestyle='--', label='Rata-rata')",
    "axes[0].legend(); axes[0].tick_params(axis='x', rotation=25)",
    "",
    "weights = [total/(len(cls_list)*c) for c in cnt_list]",
    "bars2 = axes[1].bar(short, weights, color=palette, edgecolor='white', alpha=0.85)",
    "[axes[1].text(b.get_x()+b.get_width()/2, b.get_height()+0.005, f'{w:.2f}x',",
    "              ha='center', fontsize=9, fontweight='bold') for b, w in zip(bars2, weights)]",
    "axes[1].set_title('Class Weight untuk CrossEntropyLoss', fontweight='bold')",
    "axes[1].tick_params(axis='x', rotation=25)",
    "",
    "plt.suptitle('EDA 8 – Class Imbalance', fontsize=13, fontweight='bold')",
    "plt.tight_layout(); plt.savefig('eda8_imbalance.png', dpi=100); plt.show()",
]))

# ════════════════════════════════════════════════════════════════════════
# PREPROCESSING FINAL
cells.append(md([
    "## Preprocessing Final (Berdasarkan EDA 1-8)\n\n",
    "| Prioritas | Masalah | Tindakan |\n",
    "|-----------|---------|----------|\n",
    "| 🔴 KRITIS | Duplikat beda kelas | Hapus duplikat |\n",
    "| 🟠 TINGGI | Class imbalance | WeightedSampler + class weights |\n",
    "| 🟡 SEDANG | Aspect ratio bervariasi | PadToSquare sebelum resize |\n",
    "| 🟡 SEDANG | 16 file < 10KB | Filter filesize |\n",
    "| 🟢 RENDAH | Brightness bervariasi | ColorJitter (sudah di augmentasi) |\n",
]))
cells.append(code([
    "# Step 1: Hapus duplikat beda kelas (MD5 hash)",
    "print('Menghitung hash duplikat...')",
    "hash_records = []",
    "for cls in classes:",
    "    folder = os.path.join(train_path, cls)",
    "    if not os.path.isdir(folder): continue",
    "    for fname in os.listdir(folder):",
    "        if not fname.lower().endswith(('.jpg','.jpeg','.png')): continue",
    "        fpath = os.path.join(folder, fname)",
    "        with open(fpath, 'rb') as f:",
    "            h = hashlib.md5(f.read()).hexdigest()",
    "        hash_records.append({'hash': h, 'class': cls, 'path': fpath})",
    "",
    "hash_df = pd.DataFrame(hash_records)",
    "dup_hashes = hash_df.groupby('hash')['class'].nunique()",
    "cross_hashes = dup_hashes[dup_hashes > 1].index",
    "cross_dups   = hash_df[hash_df['hash'].isin(cross_hashes)]",
    "print(f'Duplikat beda kelas: {len(cross_dups)} gambar dari {len(cross_hashes)} grup')",
    "",
    "CLASS_PRIORITY = ['realperson','fake_printed','fake_screen','fake_mask','fake_mannequin','fake_unknown']",
    "to_remove = set()",
    "for h, grp in cross_dups.groupby('hash'):",
    "    grp = grp.copy()",
    "    grp['pri'] = grp['class'].map({c:i for i,c in enumerate(CLASS_PRIORITY)}).fillna(99)",
    "    for path in grp.sort_values('pri')['path'].values[1:]:",
    "        to_remove.add(path)",
    "",
    "print(f'File akan dihapus: {len(to_remove)}')",
    "CONFIRM = True",
    "if CONFIRM:",
    "    [os.remove(p) for p in to_remove if os.path.exists(p)]",
    "    print(f'✓ {len(to_remove)} duplikat dihapus!')",
    "",
    "# Step 2: Filter file < 10KB",
    "MIN_KB = 10",
    "removed_lq = 0",
    "for cls in classes:",
    "    folder = os.path.join(train_path, cls)",
    "    if not os.path.isdir(folder): continue",
    "    for fname in os.listdir(folder):",
    "        fpath = os.path.join(folder, fname)",
    "        if os.path.getsize(fpath)/1024 < MIN_KB:",
    "            os.remove(fpath); removed_lq += 1",
    "print(f'✓ {removed_lq} file < {MIN_KB}KB dihapus!')",
    "",
    "# Rebuild df bersih",
    "data2 = []",
    "for cls in classes:",
    "    folder = os.path.join(train_path, cls)",
    "    if not os.path.isdir(folder): continue",
    "    for fname in os.listdir(folder):",
    "        if fname.lower().endswith(('.jpg','.jpeg','.png')):",
    "            data2.append({'path': os.path.join(folder, fname), 'label': cls})",
    "",
    "df_clean = pd.DataFrame(data2)",
    "df_clean['label_idx'] = df_clean['label'].map(label2idx)",
    "print(f'\\n=== Dataset Bersih ===')",
    "print(df_clean['label'].value_counts())",
    "print(f'Total: {len(df_clean)} gambar')",
]))

# ════════════════════════════════════════════════════════════════════════
# DATASET & TRANSFORM
cells.append(md("## Dataset Class & Augmentasi"))
cells.append(code([
    "class PadToSquare:",
    "    def __call__(self, img):",
    "        w, h = img.size; m = max(w,h)",
    "        pl=(m-w)//2; pr=m-w-pl; pt=(m-h)//2; pb=m-h-pt",
    "        return TF.pad(img, (pl,pt,pr,pb), 0, 'constant')",
    "",
    "class FaceDataset(Dataset):",
    "    def __init__(self, df, transform=None):",
    "        self.df=df.reset_index(drop=True); self.transform=transform",
    "    def __len__(self): return len(self.df)",
    "    def __getitem__(self, idx):",
    "        row = self.df.iloc[idx]",
    "        img = Image.open(row['path']).convert('RGB')",
    "        if self.transform: img = self.transform(img)",
    "        return img, row['label_idx']",
    "",
    "IMG_SIZE = 224",
    "TRAIN_TF = transforms.Compose([",
    "    PadToSquare(),",
    "    transforms.Resize((IMG_SIZE, IMG_SIZE)),",
    "    transforms.RandomHorizontalFlip(),",
    "    transforms.RandomRotation(15),",
    "    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),",
    "    transforms.RandomAffine(degrees=0, translate=(0.1,0.1)),",
    "    transforms.ToTensor(),",
    "    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),",
    "    transforms.RandomErasing(p=0.3),",
    "])",
    "VAL_TF = transforms.Compose([",
    "    PadToSquare(),",
    "    transforms.Resize((IMG_SIZE, IMG_SIZE)),",
    "    transforms.ToTensor(),",
    "    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),",
    "])",
    "print('Transforms siap!')",
]))

# ════════════════════════════════════════════════════════════════════════
# TRAINING
cells.append(md([
    "## Training: 5-Fold dengan EfficientNet-B0\n",
    "> **Catatan Windows/Jupyter:** `num_workers=0` wajib dipakai! ",
    "`num_workers > 0` akan menyebabkan training **freeze selamanya** karena ",
    "Windows tidak mendukung multiprocessing fork di dalam Jupyter.\n",
]))
cells.append(code([
    "def train_one_fold(df, train_idx, val_idx, fold_num):",
    "    print(f'\\n===== FOLD {fold_num} =====')",
    "    train_df = df.iloc[train_idx]",
    "    val_df   = df.iloc[val_idx]",
    "",
    "    train_ds = FaceDataset(train_df, TRAIN_TF)",
    "    val_ds   = FaceDataset(val_df,   VAL_TF)",
    "",
    "    class_counts = Counter(train_df['label_idx'].values)",
    "    weights  = [1.0/class_counts[l] for l in train_df['label_idx'].values]",
    "    sampler  = torch.utils.data.WeightedRandomSampler(weights, len(weights))",
    "",
    "    # PENTING: num_workers=0 untuk Windows + Jupyter (hindari freeze!)",
    "    train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=0, pin_memory=False)",
    "    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False,   num_workers=0, pin_memory=False)",
    "",
    "    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=6).to(device)",
    "",
    "    total = sum(class_counts.values())",
    "    cw = torch.tensor([total/(6*class_counts[i]) for i in range(6)], dtype=torch.float).to(device)",
    "    criterion = nn.CrossEntropyLoss(weight=cw)",
    "    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)",
    "    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)",
    "",
    "    best_f1, best_state = 0, None",
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
    "        preds, trues = [], []",
    "        with torch.no_grad():",
    "            for imgs, labels in val_loader:",
    "                p = model(imgs.to(device)).argmax(1).cpu().numpy()",
    "                preds.extend(p); trues.extend(labels.numpy())",
    "",
    "        f1 = f1_score(trues, preds, average='macro')",
    "        print(f'  Epoch {epoch+1:2d} | Loss: {train_loss/len(train_loader):.4f} | Macro F1: {f1:.4f}')",
    "        if f1 > best_f1:",
    "            best_f1 = f1; best_state = model.state_dict().copy()",
    "",
    "    print(f'  >> Best F1 Fold {fold_num}: {best_f1:.4f}')",
    "    model.load_state_dict(best_state)",
    "    return model, best_f1",
    "",
    "# Gunakan df_clean (setelah preprocessing) jika sudah dijalankan,",
    "# atau df jika melewati preprocessing",
    "df_train = df_clean if 'df_clean' in dir() and len(df_clean) > 0 else df",
    "",
    "skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
    "models, scores = [], []",
    "for fold, (train_idx, val_idx) in enumerate(skf.split(df_train, df_train['label_idx'])):",
    "    m, s = train_one_fold(df_train, train_idx, val_idx, fold+1)",
    "    models.append(m); scores.append(s)",
    "",
    "print(f'\\n===== HASIL AKHIR =====')",
    "print(f'Average Macro F1 : {sum(scores)/len(scores):.4f}')",
    "for i, s in enumerate(scores): print(f'  Fold {i+1}: {s:.4f}')",
]))

# ════════════════════════════════════════════════════════════════════════
# CONFUSION MATRIX
cells.append(md("## Confusion Matrix & Classification Report (Ensemble)"))
cells.append(code([
    "folds = list(skf.split(df_train, df_train['label_idx']))",
    "_, val_idx = folds[-1]",
    "val_df  = df_train.iloc[val_idx]",
    "val_ds  = FaceDataset(val_df, VAL_TF)",
    "val_ldr = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)",
    "",
    "all_probs = []",
    "for m in models:",
    "    m.eval(); fp = []",
    "    with torch.no_grad():",
    "        for imgs, _ in val_ldr:",
    "            fp.append(torch.softmax(m(imgs.to(device)),1).cpu().numpy())",
    "    all_probs.append(np.concatenate(fp))",
    "",
    "avg_probs = np.mean(all_probs, axis=0)",
    "preds     = np.argmax(avg_probs, axis=1)",
    "trues     = val_df['label_idx'].values",
    "names     = [idx2label[i] for i in range(6)]",
    "",
    "print(classification_report(trues, preds, target_names=names))",
    "",
    "cm = confusion_matrix(trues, preds)",
    "fig, ax = plt.subplots(figsize=(10,8))",
    "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=names, yticklabels=names)",
    "plt.title('Confusion Matrix – Ensemble', fontsize=14, fontweight='bold')",
    "plt.xlabel('Predicted'); plt.ylabel('Actual')",
    "plt.tight_layout()",
    "plt.savefig('confusion_matrix.png', dpi=150)",
    "plt.show()",
]))

# ════════════════════════════════════════════════════════════════════════
# SUBMISSION
cells.append(md("## Inferensi & Generate submission.csv"))
cells.append(code([
    "test_files = sorted([f for f in os.listdir(test_path)",
    "                     if f.lower().endswith(('.jpg','.jpeg','.png'))])",
    "test_data  = [{'path': os.path.join(test_path,f), 'label': 'unknown',",
    "               'label_idx': 0, 'id': os.path.splitext(f)[0]} for f in test_files]",
    "test_df    = pd.DataFrame(test_data)",
    "",
    "class TestDS(Dataset):",
    "    def __init__(self, df, tf): self.df=df.reset_index(drop=True); self.tf=tf",
    "    def __len__(self): return len(self.df)",
    "    def __getitem__(self, i):",
    "        return self.tf(Image.open(self.df.iloc[i]['path']).convert('RGB'))",
    "",
    "test_ldr = DataLoader(TestDS(test_df, VAL_TF), batch_size=64,",
    "                       shuffle=False, num_workers=0)",
    "",
    "all_test = []",
    "for m in models:",
    "    m.eval(); fp = []",
    "    with torch.no_grad():",
    "        for imgs in test_ldr:",
    "            fp.append(torch.softmax(m(imgs.to(device)),1).cpu().numpy())",
    "    all_test.append(np.concatenate(fp))",
    "",
    "test_preds = np.argmax(np.mean(all_test, axis=0), axis=1)",
    "submission = pd.DataFrame({",
    "    'id':    test_df['id'],",
    "    'label': [idx2label[p] for p in test_preds]",
    "})",
    "submission.to_csv('submission.csv', index=False)",
    "print(f'Submission disimpan! ({len(submission)} baris)')",
    "print(submission['label'].value_counts())",
    "submission.head(10)",
]))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (FindIT DAC)",
            "language": "python",
            "name": "codegeex-agent"
        },
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

print(f"Notebook berhasil dibuat! Total: {len(cells)} cells")
