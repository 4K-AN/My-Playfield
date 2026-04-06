import json

with open("e:/GIT/My-Playfield/Lomba/Findit/dac_find_it_2026.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

def src(*lines):
    result = [line + '\n' for line in lines]
    if result:
        result[-1] = result[-1].rstrip('\n')
    return result

eda6_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## EDA 6: Analisis Tekstur & Frekuensi (FFT)\n",
            "\n",
            "**Tujuan:** Mendeteksi pola tersembunyi di domain frekuensi yang tidak terlihat di domain piksel biasa.\n",
            "\n",
            "- `fake_printed` → Moiré pattern dari pola halftone cetakan\n",
            "- `fake_screen` → Pola grid piksel layar (reguler, terstruktur)\n",
            "- `realperson` → Tidak ada pola reguler, frekuensi acak/organik\n",
            "- `fake_mask` → Mungkin ada pola kilap permukaan\n",
            "\n",
            "**Mengapa ini penting?** Jika pola frekuensi antar kelas berbeda secara signifikan,\n",
            "model CNN akan otomatis mempelajarinya melalui filter konvolusinya.\n",
            "Namun kalau kita tahu ini eksplisit, kita bisa memvalidasi apakah arsitektur kita sudah cukup 'dalam' untuk menangkap tekstur frekuensi tinggi tersebut."
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
            "import matplotlib.pyplot as plt",
            "from PIL import Image",
            "",
            "train_path = './dataset/train'",
            "classes    = sorted(os.listdir(train_path))",
            "RESIZE     = (256, 256)  # Ukuran seragam untuk FFT",
            "",
            "fig, axes = plt.subplots(len(classes), 3, figsize=(15, len(classes) * 3))",
            "",
            "for row, cls in enumerate(classes):",
            "    folder = os.path.join(train_path, cls)",
            "    files  = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg','.jpeg','.png'))]",
            "    fname  = random.choice(files)",
            "    fpath  = os.path.join(folder, fname)",
            "",
            "    img  = Image.open(fpath).convert('RGB').resize(RESIZE)",
            "    gray = np.array(img.convert('L'), dtype=np.float32)",
            "",
            "    # FFT 2D",
            "    fft      = np.fft.fft2(gray)",
            "    fft_shift = np.fft.fftshift(fft)          # Pindahkan frekuensi DC ke tengah",
            "    magnitude = np.log1p(np.abs(fft_shift))    # Log scale agar kontras terlihat",
            "",
            "    # Radial power spectrum (rata-rata energi pada tiap radius frekuensi)",
            "    h, w  = magnitude.shape",
            "    cy, cx = h // 2, w // 2",
            "    Y, X  = np.ogrid[:h, :w]",
            "    radius = np.sqrt((X - cx)**2 + (Y - cy)**2).astype(int)",
            "    max_r  = min(cx, cy)",
            "    power  = np.array([magnitude[radius == r].mean() if (radius == r).any() else 0",
            "                       for r in range(max_r)])",
            "",
            "    # Kolom 1: Gambar asli",
            "    axes[row][0].imshow(img)",
            "    axes[row][0].set_title(f'{cls}\\nOriginal', fontsize=9)",
            "    axes[row][0].axis('off')",
            "",
            "    # Kolom 2: FFT magnitude (spektrum frekuensi)",
            "    axes[row][1].imshow(magnitude, cmap='inferno')",
            "    axes[row][1].set_title('FFT Magnitude Spectrum', fontsize=9)",
            "    axes[row][1].axis('off')",
            "",
            "    # Kolom 3: Radial power spectrum",
            "    axes[row][2].plot(power, color='dodgerblue', linewidth=1.5)",
            "    axes[row][2].set_title('Radial Power Spectrum', fontsize=9)",
            "    axes[row][2].set_xlabel('Frekuensi (radius)')",
            "    axes[row][2].set_ylabel('Log Power')",
            "    axes[row][2].grid(alpha=0.3)",
            "",
            "plt.suptitle('EDA 6 – Analisis Tekstur & Frekuensi (FFT) per Kelas',",
            "             fontsize=14, fontweight='bold', y=1.01)",
            "plt.tight_layout()",
            "plt.savefig('eda6_fft_analysis.png', dpi=120, bbox_inches='tight')",
            "plt.show()",
            "print('Plot disimpan sebagai eda6_fft_analysis.png')"
        )
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["### EDA 6b: Rata-rata Spektrum FFT per Kelas (50 Sampel)"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src(
            "# Rata-rata radial power spectrum dari 50 sampel per kelas",
            "# Ini lebih representatif daripada 1 gambar saja",
            "N_SAMPLE = 50",
            "RESIZE   = (256, 256)",
            "max_r    = RESIZE[0] // 2",
            "",
            "fig, ax = plt.subplots(figsize=(12, 6))",
            "palette = ['#4CAF50','#FF5722','#2196F3','#9C27B0','#FF9800','#607D8B']",
            "",
            "for ci, cls in enumerate(classes):",
            "    folder = os.path.join(train_path, cls)",
            "    files  = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg','.jpeg','.png'))]",
            "    sample = random.sample(files, min(N_SAMPLE, len(files)))",
            "",
            "    all_power = []",
            "    for fname in sample:",
            "        try:",
            "            img  = Image.open(os.path.join(folder, fname)).convert('L').resize(RESIZE)",
            "            gray = np.array(img, dtype=np.float32)",
            "            fft_shift = np.fft.fftshift(np.fft.fft2(gray))",
            "            mag  = np.log1p(np.abs(fft_shift))",
            "            h, w = mag.shape",
            "            cy, cx = h//2, w//2",
            "            Y, X = np.ogrid[:h, :w]",
            "            rad  = np.sqrt((X-cx)**2 + (Y-cy)**2).astype(int)",
            "            pwr  = np.array([mag[rad==r].mean() if (rad==r).any() else 0 for r in range(max_r)])",
            "            all_power.append(pwr)",
            "        except:",
            "            pass",
            "",
            "    if all_power:",
            "        mean_power = np.mean(all_power, axis=0)",
            "        ax.plot(mean_power, label=cls, color=palette[ci], linewidth=2, alpha=0.85)",
            "",
            "ax.set_xlabel('Frekuensi Spatial (radius dari DC)', fontsize=11)",
            "ax.set_ylabel('Rata-rata Log Power', fontsize=11)",
            "ax.set_title(f'EDA 6b – Rata-rata Radial Power Spectrum per Kelas ({N_SAMPLE} sampel)',",
            "             fontsize=13, fontweight='bold')",
            "ax.legend(fontsize=10)",
            "ax.grid(alpha=0.3)",
            "plt.tight_layout()",
            "plt.savefig('eda6b_avg_power_spectrum.png', dpi=120)",
            "plt.show()"
        )
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Cara Membaca Hasil EDA 6\n",
            "\n",
            "**FFT Magnitude Spectrum (kolom tengah):**\n",
            "- Titik cerah di tengah = energi frekuensi rendah (global brightness)\n",
            "- Garis/pola terang yang menjalar ke pinggir = ada **pola berulang** (Moiré/piksel grid)\n",
            "- `fake_printed` → Lihat apakah ada titik-titik terang tersusun rapi (halftone pattern)\n",
            "- `fake_screen` → Lihat apakah ada pola silang/grid yang simetris\n",
            "- `realperson` → Biasanya hanya titik terang di tengah, menyebar organik\n",
            "\n",
            "**Radial Power Spectrum (kolom kanan):**\n",
            "- Kurva turun cepat = dominan frekuensi rendah (gambar 'halus')\n",
            "- Ada 'bump'/tonjolan di frekuensi menengah-tinggi = ada pola tekstur berulang\n",
            "- `fake_printed` & `fake_screen` diharapkan punya bump di frekuensi menengah\n",
            "\n",
            "**Implikasi bagi Model:**\n",
            "- Jika pola frekuensi antar kelas berbeda → arsitektur shallow (EfficientNet-B0) sudah cukup\n",
            "- Jika pola sangat mirip → pertimbangkan backbone yang lebih dalam (B3/B4) atau tambahan attention mechanism"
        ]
    }
]

# Sisipkan sebelum cell training
insert_idx = len(nb["cells"])
for i, cell in enumerate(nb["cells"]):
    src_text = "".join(cell.get("source", []))
    if "train_one_fold" in src_text or "StratifiedKFold" in src_text:
        insert_idx = i
        break

for j, cell in enumerate(eda6_cells):
    nb["cells"].insert(insert_idx + j, cell)

with open("e:/GIT/My-Playfield/Lomba/Findit/dac_find_it_2026.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print(f"EDA 6 ditambahkan di posisi index {insert_idx}. Total cells: {len(nb['cells'])}")
