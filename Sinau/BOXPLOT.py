import matplotlib.pyplot as plt
import numpy as np

# Data Nilai Kalkulus
data = np.array([86, 85, 81, 81, 81, 90, 89, 86, -20, 84, 85, 80, 84, 86, 89])

# Membuat Boxplot
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(8, 6))

# Mengatur properti untuk tampilan outlier yang lebih jelas
outlier_props = dict(markerfacecolor='r', marker='D', markersize=8)

ax.boxplot(data, vert=True, patch_artist=True, whis=1.5,
           medianprops=dict(color='black', linewidth=2),
           flierprops=outlier_props)

# Judul dan Label
ax.set_title('Boxplot Nilai Kalkulus', fontsize=16)
ax.set_ylabel('Nilai', fontsize=12)
ax.set_xticklabels(['Kalkulus'])
ax.grid(True)

# Menampilkan plot
plt.show()