import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# --- Parameter Awal Sesuai Materi di Canvas ---
pv = 1_000_000  # Present Value (Modal Awal dalam Rupiah)
r = 0.10         # Suku bunga tahunan (10%)
n_years = 20     # Jumlah tahun investasi
t = np.arange(0, n_years + 1, 1) # Array waktu dari tahun 0 sampai 20

# --- Perhitungan Future Value untuk Setiap Frekuensi Compounding ---

# 1. Tahunan (k=1)
fv_tahunan = pv * (1 + r)**t

# 2. Semi-Annual / 6 Bulanan (k=2)
fv_semi_annual = pv * (1 + r/2)**(2*t)

# 3. Bulanan (k=12)
fv_bulanan = pv * (1 + r/12)**(12*t)

# 4. Harian (k=365)
# Dihitung untuk menunjukkan perbedaan yang lebih halus
fv_harian = pv * (1 + r/365)**(365*t)

# 5. Continuous Compounding (Batas Maksimum)
fv_continuous = pv * np.exp(r * t)

# --- Membuat Grafik (Visualisasi) ---
plt.style.use('seaborn-v0_8-whitegrid') # Menggunakan style agar grafik terlihat bagus
fig, ax = plt.subplots(figsize=(12, 8))

# Plot setiap garis pertumbuhan
ax.plot(t, fv_tahunan, label='Tahunan (Annual)', linestyle='--')
ax.plot(t, fv_semi_annual, label='6 Bulanan (Semi-Annual)', linestyle='-.')
ax.plot(t, fv_bulanan, label='Bulanan (Monthly)', linestyle=':')
ax.plot(t, fv_continuous, label='Kontinu (Continuous)', linewidth=2.5, color='black')

# --- Pengaturan Tampilan Grafik ---
ax.set_title('Visualisasi Pertumbuhan Compounding', fontsize=16, fontweight='bold')
ax.set_xlabel('Waktu (Tahun)', fontsize=12)
ax.set_ylabel('Future Value (Rp)', fontsize=12)
ax.legend(title='Frekuensi Compounding', fontsize=10)
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Format sumbu Y agar menampilkan format Rupiah yang mudah dibaca
formatter = mticker.FuncFormatter(lambda x, p: 'Rp {:,.0f}'.format(x).replace(',', '.'))
ax.yaxis.set_major_formatter(formatter)
plt.xticks(np.arange(0, n_years + 1, 2)) # Menampilkan label tahun setiap 2 tahun

# Menampilkan grafik
plt.tight_layout()
plt.show()

