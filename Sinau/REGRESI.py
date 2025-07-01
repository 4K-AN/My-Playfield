import matplotlib.pyplot as plt
import numpy as np

# --- Data Awal ---
# Data X (Nilai ProbStat) dengan nilai yang sudah diisi
x_data = np.array([86, 70, 90, 76, 79, 80.5555556, 76, 90, 72, 86])
# Data Y (Nilai MatDis)
y_data = np.array([90, 73, 73, 78, 86, 84, 74, 20, 80, 72])

# Koordinat outlier
outlier_x = 90
outlier_y = 20

# --- Persamaan Garis Regresi ---
# 1. Garis regresi dengan outlier (garis merah)
# y = 163.01 - 1.117x
x_line__outlier = np.linspace(min(x_data), max(x_data), 100)
y_line_outlier = 163.001059 - 1.11725453 * x_line__outlier

# 2. Garis regresi tanpa outlier (garis hijau)
# y = 71.26 + 0.096x
x_line_corrected = np.linspace(min(x_data), max(x_data), 100)
y_line_corrected = 71.24741 + 0.096112 * x_line_corrected

# --- Membuat Grafik ---
plt.style.use('seaborn-v0_8-whitegrid') # Menggunakan style agar grafik terlihat bagus
fig, ax = plt.subplots(figsize=(10, 6))

# Scatter plot untuk semua data
ax.scatter(x_data, y_data, label='Data Poin', color='royalblue', zorder=5)

# Menandai titik outlier secara spesifik
ax.scatter(outlier_x, outlier_y, color='red', s=100, zorder=6, edgecolor='black', label='Outlier (90, 20)')

# Plot garis regresi merah (dengan outlier)
ax.plot(x_line__outlier, y_line_outlier, color='red', linestyle='--', label='Garis Regresi dengan Outlier')

# Plot garis regresi hijau (setelah perbaikan)
ax.plot(x_line_corrected, y_line_corrected, color='green', label='Garis Regresi yang Benar')

# --- Label dan Judul ---
ax.set_title('Perbandingan Garis Regresi Sebelum dan Sesudah Menangani Outlier', fontsize=16)
ax.set_xlabel('Nilai ProbStat (X)', fontsize=12)
ax.set_ylabel('Nilai MatDis (Y)', fontsize=12)
ax.legend()
ax.grid(True)

# Menampilkan grafik
plt.show()