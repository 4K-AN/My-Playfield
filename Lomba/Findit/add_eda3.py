import json

with open("e:/GIT/My-Playfield/Lomba/Findit/dac_find_it_2026.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

def src(*lines):
    result = [line + '\n' for line in lines]
    if result:
        result[-1] = result[-1].rstrip('\n')
    return result

eda3_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## EDA 3: Visual Inspection & Artifact Analysis\n",
            "\n",
            "Tujuan: Melihat langsung ciri khas visual tiap kelas untuk menentukan strategi augmentasi.\n",
            "\n",
            "| Kelas | Ekspektasi Artefak |\n",
            "|-------|--------------------|\n",
            "| `realperson` | Foto natural, berbagai pencahayaan |\n",
            "| `fake_printed` | Tekstur kertas, Moiré pattern, tepian kertas |\n",
            "| `fake_screen` | Piksel layar, glare/pantulan cahaya, warna over-saturated |\n",
            "| `fake_mask` | Kilap plastik/silikon, batas masker di tepi wajah |\n",
            "| `fake_mannequin` | Statis/tidak natural, pencahayaan terlalu merata |\n",
            "| `fake_unknown` | Campuran artefak tidak diketahui |\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src(
            "import os, random",
            "from PIL import Image",
            "import matplotlib.pyplot as plt",
            "import matplotlib.patches as mpatches",
            "import numpy as np",
            "",
            "train_path = './dataset/train'",
            "classes    = sorted(os.listdir(train_path))",
            "N_SAMPLES  = 4  # Jumlah sampel per kelas",
            "",
            "fig, axes = plt.subplots(len(classes), N_SAMPLES, figsize=(16, len(classes) * 3))",
            "",
            "for row, cls in enumerate(classes):",
            "    folder = os.path.join(train_path, cls)",
            "    files  = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg','.jpeg','.png'))]",
            "    samples = random.sample(files, min(N_SAMPLES, len(files)))",
            "",
            "    for col, fname in enumerate(samples):",
            "        img_path = os.path.join(folder, fname)",
            "        img = Image.open(img_path).convert('RGB')",
            "        w, h = img.size",
            "        fsize_kb = os.path.getsize(img_path) / 1024",
            "        ar = w / h",
            "",
            "        axes[row][col].imshow(img)",
            "        axes[row][col].axis('off')",
            "",
            "        # Warna judul: merah jika filesize < 15KB (low quality), hijau jika OK",
            "        color = 'red' if fsize_kb < 15 else 'green'",
            "        axes[row][col].set_title(",
            "            f'{w}x{h} | AR:{ar:.2f} | {fsize_kb:.0f}KB',",
            "            fontsize=8, color=color",
            "        )",
            "",
            "    # Label kelas di sisi kiri",
            "    axes[row][0].set_ylabel(cls, fontsize=11, fontweight='bold', rotation=90, labelpad=5)",
            "",
            "red_patch   = mpatches.Patch(color='red',   label='Filesize < 15KB (low quality)')",
            "green_patch = mpatches.Patch(color='green', label='Filesize OK')",
            "fig.legend(handles=[green_patch, red_patch], loc='lower center', ncol=2, fontsize=10)",
            "",
            "plt.suptitle('EDA 3 – Visual Inspection per Kelas\\n(Width x Height | Aspect Ratio | Filesize)',",
            "             fontsize=14, fontweight='bold', y=1.01)",
            "plt.tight_layout()",
            "plt.savefig('eda3_visual_inspection.png', dpi=120, bbox_inches='tight')",
            "plt.show()",
            "print('Plot disimpan sebagai eda3_visual_inspection.png')"
        )
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### EDA 3b: Distribusi Aspect Ratio per Kelas\n",
            "Visualisasi seberapa beragam aspect ratio gambar di setiap kelas."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src(
            "import matplotlib.pyplot as plt",
            "import os",
            "from PIL import Image",
            "",
            "train_path = './dataset/train'",
            "classes    = sorted(os.listdir(train_path))",
            "",
            "fig, axes = plt.subplots(2, 3, figsize=(15, 8))",
            "axes = axes.flatten()",
            "",
            "for i, cls in enumerate(classes):",
            "    folder = os.path.join(train_path, cls)",
            "    aspect_ratios = []",
            "    for fname in os.listdir(folder):",
            "        if fname.lower().endswith(('.jpg','.jpeg','.png')):",
            "            try:",
            "                w, h = Image.open(os.path.join(folder, fname)).size",
            "                aspect_ratios.append(round(w / h, 2))",
            "            except:",
            "                pass",
            "",
            "    axes[i].hist(aspect_ratios, bins=20, color='steelblue', edgecolor='white', alpha=0.85)",
            "    axes[i].axvline(1.0, color='red', linestyle='--', linewidth=1.5, label='Square (AR=1)')",
            "    axes[i].set_title(f'{cls}\\nmean AR={sum(aspect_ratios)/len(aspect_ratios):.2f}', fontweight='bold')",
            "    axes[i].set_xlabel('Aspect Ratio (W/H)')",
            "    axes[i].set_ylabel('Jumlah Gambar')",
            "    axes[i].legend(fontsize=8)",
            "",
            "plt.suptitle('Distribusi Aspect Ratio per Kelas', fontsize=14, fontweight='bold')",
            "plt.tight_layout()",
            "plt.savefig('eda3b_aspect_ratio.png', dpi=120)",
            "plt.show()",
            "print('Plot disimpan sebagai eda3b_aspect_ratio.png')"
        )
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### EDA 3c: Identifikasi Gambar Low-Quality (< 15 KB)\n",
            "Berdasarkan EDA 1, ada gambar dengan filesize sangat kecil yang berpotensi mengganggu training."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src(
            "import os",
            "import pandas as pd",
            "",
            "train_path = './dataset/train'",
            "THRESHOLD_KB = 15",
            "",
            "low_quality = []",
            "for cls in sorted(os.listdir(train_path)):",
            "    folder = os.path.join(train_path, cls)",
            "    for fname in os.listdir(folder):",
            "        fpath = os.path.join(folder, fname)",
            "        fsize = os.path.getsize(fpath) / 1024",
            "        if fsize < THRESHOLD_KB:",
            "            low_quality.append({'class': cls, 'file': fname, 'size_kb': round(fsize, 2)})",
            "",
            "lq_df = pd.DataFrame(low_quality)",
            "if len(lq_df) > 0:",
            "    print(f'Total gambar < {THRESHOLD_KB}KB: {len(lq_df)}')",
            "    print('\\nPer kelas:')",
            "    print(lq_df['class'].value_counts())",
            "    print('\\nContoh gambar low quality:')",
            "    print(lq_df.head(10))",
            "else:",
            "    print(f'Tidak ada gambar dengan ukuran < {THRESHOLD_KB}KB.')",
            "",
            "# Persentase low-quality",
            "total_imgs = sum(len(os.listdir(os.path.join(train_path, c))) for c in os.listdir(train_path))",
            "pct = len(lq_df) / total_imgs * 100 if total_imgs > 0 else 0",
            "print(f'\\nPersentase low-quality: {pct:.2f}% dari total {total_imgs} gambar')",
            "print('\\nKesimpulan:')",
            "if pct < 2:",
            "    print('  → Jumlah sangat sedikit (<2%), AMAN untuk dibiarkan. Model tidak akan terganggu.')",
            "elif pct < 5:",
            "    print('  → Relatif kecil (2-5%), pertimbangkan filter jika F1 score stagnan.')",
            "else:",
            "    print('  → Cukup banyak (>5%), REKOMENDASIKAN difilter sebelum training.')"
        )
    }
]

# Temukan posisi cell terakhir sebelum training/super train cell dan insert EDA 3 di situ
# Atau tambahkan di akhir saja (sebelum cell training jika ada, atau di akhir)
insert_idx = len(nb["cells"])  # default: akhir

# Cari apakah ada cell training dan sisipkan EDA 3 sebelumnya
for i, cell in enumerate(nb["cells"]):
    src_text = "".join(cell.get("source", []))
    if "train_one_fold" in src_text or "===== FOLD" in src_text or "StratifiedKFold" in src_text:
        insert_idx = i
        break

for j, cell in enumerate(eda3_cells):
    nb["cells"].insert(insert_idx + j, cell)

with open("e:/GIT/My-Playfield/Lomba/Findit/dac_find_it_2026.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("EDA 3 cells berhasil ditambahkan ke notebook!")
