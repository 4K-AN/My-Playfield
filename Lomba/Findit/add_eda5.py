import json

with open("e:/GIT/My-Playfield/Lomba/Findit/dac_find_it_2026.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

def src(*lines):
    result = [line + '\n' for line in lines]
    if result:
        result[-1] = result[-1].rstrip('\n')
    return result

eda5_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## EDA 5: Analisis Kecerahan, Warna & Kejelasan (Sharpness)\n",
            "\n",
            "**Tujuan:**\n",
            "1. **Brightness/Luminance** — Apakah ada kelas yang lebih gelap/terang? (fake_screen biasanya over-exposed)\n",
            "2. **Color Channel Mean** — Apakah ada bias warna antar kelas? (fake_mask mungkin lebih 'dingin' vs realperson)\n",
            "3. **Sharpness Score (Laplacian Variance)** — Membantu memutuskan apakah 16 file kecil dari EDA 4 terlalu blur untuk dipakai\n",
            "\n",
            "> **Konteks dari EDA 4:** 15 dari 16 file < 10KB adalah `fake_mannequin`. \n",
            "> EDA 5 ini akan membuktikan secara kuantitatif apakah file-file tersebut memang terlalu blur."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src(
            "import os, random",
            "import numpy as np",
            "import pandas as pd",
            "import matplotlib.pyplot as plt",
            "from PIL import Image",
            "import cv2",
            "",
            "train_path = './dataset/train'",
            "MAX_PER_CLASS = 150  # Sample acak untuk efisiensi",
            "",
            "results = []",
            "for cls in sorted(os.listdir(train_path)):",
            "    folder = os.path.join(train_path, cls)",
            "    files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg','.jpeg','.png'))]",
            "    sample = random.sample(files, min(MAX_PER_CLASS, len(files)))",
            "",
            "    for fname in sample:",
            "        fpath = os.path.join(folder, fname)",
            "        fsize_kb = os.path.getsize(fpath) / 1024",
            "        try:",
            "            img_pil = Image.open(fpath).convert('RGB')",
            "            img_np  = np.array(img_pil)",
            "",
            "            # Brightness (mean luminance via grayscale)",
            "            gray = np.mean(img_np) / 255.0",
            "",
            "            # Color channel means (normalized 0-1)",
            "            r_mean = img_np[:,:,0].mean() / 255.0",
            "            g_mean = img_np[:,:,1].mean() / 255.0",
            "            b_mean = img_np[:,:,2].mean() / 255.0",
            "",
            "            # Sharpness: Laplacian Variance (tinggi = tajam, rendah = blur)",
            "            gray_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)",
            "            sharpness = cv2.Laplacian(gray_cv, cv2.CV_64F).var()",
            "",
            "            results.append({",
            "                'class': cls,",
            "                'file':  fname,",
            "                'size_kb': fsize_kb,",
            "                'brightness': gray,",
            "                'r_mean': r_mean,",
            "                'g_mean': g_mean,",
            "                'b_mean': b_mean,",
            "                'sharpness': sharpness",
            "            })",
            "        except Exception as e:",
            "            pass",
            "",
            "stats_df = pd.DataFrame(results)",
            "print(f'Berhasil dianalisis: {len(stats_df)} gambar')",
            "print(stats_df.groupby('class')[['brightness','sharpness','size_kb']].mean().round(2))"
        )
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src(
            "# ── PLOT 1: Brightness per Kelas ──────────────────────────────────────────────",
            "fig, axes = plt.subplots(1, 3, figsize=(18, 5))",
            "",
            "# Kiri: Brightness boxplot",
            "classes = sorted(stats_df['class'].unique())",
            "brightness_data = [stats_df[stats_df['class']==c]['brightness'].values for c in classes]",
            "bp = axes[0].boxplot(brightness_data, labels=[c.replace('fake_','f_') for c in classes],",
            "                     patch_artist=True, notch=False)",
            "colors = ['#4CAF50','#FF5722','#2196F3','#9C27B0','#FF9800','#607D8B']",
            "for patch, color in zip(bp['boxes'], colors):",
            "    patch.set_facecolor(color)",
            "    patch.set_alpha(0.75)",
            "axes[0].set_title('Brightness per Kelas', fontweight='bold')",
            "axes[0].set_ylabel('Mean Brightness (0=hitam, 1=putih)')",
            "axes[0].tick_params(axis='x', rotation=25)",
            "",
            "# Tengah: Color channel means",
            "x = np.arange(len(classes))",
            "w = 0.25",
            "r_vals = [stats_df[stats_df['class']==c]['r_mean'].mean() for c in classes]",
            "g_vals = [stats_df[stats_df['class']==c]['g_mean'].mean() for c in classes]",
            "b_vals = [stats_df[stats_df['class']==c]['b_mean'].mean() for c in classes]",
            "axes[1].bar(x - w, r_vals, w, label='Red',   color='#E53935', alpha=0.85)",
            "axes[1].bar(x,     g_vals, w, label='Green', color='#43A047', alpha=0.85)",
            "axes[1].bar(x + w, b_vals, w, label='Blue',  color='#1E88E5', alpha=0.85)",
            "axes[1].set_xticks(x)",
            "axes[1].set_xticklabels([c.replace('fake_','f_') for c in classes], rotation=25, ha='right')",
            "axes[1].set_title('Rata-rata Channel RGB per Kelas', fontweight='bold')",
            "axes[1].set_ylabel('Mean Value (0-1)')",
            "axes[1].legend()",
            "",
            "# Kanan: Sharpness boxplot (log scale untuk kejelasan)",
            "sharpness_data = [stats_df[stats_df['class']==c]['sharpness'].values for c in classes]",
            "bp2 = axes[2].boxplot(sharpness_data, labels=[c.replace('fake_','f_') for c in classes],",
            "                      patch_artist=True, notch=False)",
            "for patch, color in zip(bp2['boxes'], colors):",
            "    patch.set_facecolor(color)",
            "    patch.set_alpha(0.75)",
            "axes[2].set_yscale('log')",
            "axes[2].set_title('Sharpness (Laplacian Var) per Kelas\\n(log scale, lebih tinggi = lebih tajam)', fontweight='bold')",
            "axes[2].set_ylabel('Laplacian Variance (log)')",
            "axes[2].tick_params(axis='x', rotation=25)",
            "",
            "plt.suptitle('EDA 5 – Analisis Brightness, Warna & Sharpness', fontsize=14, fontweight='bold')",
            "plt.tight_layout()",
            "plt.savefig('eda5_brightness_color_sharpness.png', dpi=120)",
            "plt.show()"
        )
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src(
            "# ── Fokus: Cek Sharpness file < 10KB (temuan EDA 4) ──────────────────────────",
            "print('=== Sharpness Score untuk file < 10KB ===')",
            "low_q = stats_df[stats_df['size_kb'] < 10].sort_values('sharpness')",
            "if len(low_q) > 0:",
            "    print(low_q[['class','file','size_kb','sharpness','brightness']].to_string(index=False))",
            "    print(f'\\nSharpness rata-rata semua gambar  : {stats_df[\"sharpness\"].mean():.1f}')",
            "    print(f'Sharpness rata-rata file < 10KB   : {low_q[\"sharpness\"].mean():.1f}')",
            "    ratio = low_q['sharpness'].mean() / stats_df['sharpness'].mean()",
            "    print(f'Rasio vs rata-rata                : {ratio:.2f}x')",
            "    print()",
            "    if ratio < 0.3:",
            "        print('KESIMPULAN: File-file < 10KB jauh lebih blur (< 30% rata-rata).')",
            "        print('            REKOMENDASIKAN DIBUANG dari dataset training.')",
            "    elif ratio < 0.6:",
            "        print('KESIMPULAN: File-file < 10KB cukup blur tapi masih ada informasi.')",
            "        print('            Pertimbangkan filter jika model underperform di fake_mannequin.')",
            "    else:",
            "        print('KESIMPULAN: File-file < 10KB masih cukup tajam.')",
            "        print('            TIDAK PERLU dibuang, tetap sertakan dalam training.')",
            "else:",
            "    print('Tidak ada file < 10KB dalam sample yang dianalisis.')"
        )
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Rangkuman EDA 5\n",
            "\n",
            "| Metrik | Temuan Kunci |\n",
            "|--------|--------------|\n",
            "| **Brightness** | Apakah `fake_screen` lebih terang? `fake_mask` lebih gelap? |\n",
            "| **Color Channels** | Apakah ada bias warna signifikan antar kelas? |\n",
            "| **Sharpness** | Apakah 16 file < 10KB memang blur parah? |\n",
            "\n",
            "**Implikasi untuk Augmentasi:**\n",
            "- Jika `fake_screen` jauh lebih terang → `ColorJitter` penting untuk normalize\n",
            "- Jika sharpness file kecil < 30% rata-rata → buang dari training\n",
            "- Jika bias warna signifikan → pertimbangkan `RandomGrayscale` untuk paksa model fokus ke tekstur, bukan warna"
        ]
    }
]

# Sisipkan EDA 5 setelah EDA 3/4 (sebelum cell training)
insert_idx = len(nb["cells"])
for i, cell in enumerate(nb["cells"]):
    src_text = "".join(cell.get("source", []))
    if "train_one_fold" in src_text or "StratifiedKFold" in src_text:
        insert_idx = i
        break

for j, cell in enumerate(eda5_cells):
    nb["cells"].insert(insert_idx + j, cell)

with open("e:/GIT/My-Playfield/Lomba/Findit/dac_find_it_2026.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print(f"EDA 5 ditambahkan di posisi index {insert_idx}. Total cells: {len(nb['cells'])}")
