import numpy as np
import matplotlib.pyplot as plt

# Defining the corrected piecewise function
def f(t):
    if t <= 2:
        return 2 * t
    elif t < 6:
        return 1 / (5 * t - 10) + 2     # hyperbola segment
    elif t < 9:
        return (t - 7)**2 + 2           # quadratic segment
    else:
        return (6 * t - 66) / (t - 11)  # stabilization rational segment

# Time arrays for each segment
t1 = np.linspace(0, 2, 200)            # 0 ≤ t ≤ 2
t2 = np.linspace(2, 6, 400, endpoint=False)  # 2 < t < 6
t3 = np.linspace(6, 9, 300, endpoint=False)  # 6 ≤ t < 9
t4 = np.linspace(9, 15, 600)           # t ≥ 9 (up to 15 for visualization)

# Compute values
f1 = [f(t) for t in t1]
f2 = [f(t) for t in t2]
f3 = [f(t) for t in t3]
f4 = [f(t) for t in t4]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(t1, f1, label='0 ≤ t ≤ 2: f(t)=2t', linewidth=2)
plt.plot(t2, f2, label='2 < t < 6: f(t)=1/(5t−10)+2', linewidth=2)
plt.plot(t3, f3, label='6 ≤ t < 9: f(t)=(t−7)²+2', linewidth=2)
plt.plot(t4, f4, label='t ≥ 9: f(t)=(6t−66)/(t−11)', linewidth=2)

# Mark transition points
# t=2,6,9 with closed/open circlesimport numpy as np
import matplotlib.pyplot as plt

# Membuat area plot
fig, ax = plt.subplots(figsize=(10, 8))

# 1. Segmen 1: Garis lurus [0, 2]
t1 = np.linspace(0, 2, 100)
y1 = 2 * t1
ax.plot(t1, y1, color='blue', linewidth=2, label='f(t) = 2t')
ax.plot(2, 4, 'o', color='blue', markersize=8) # Titik solid

# 2. Segmen 2: Hiperbola dengan asimtot (2, 6)
# Kita mulai t2 sedikit setelah 2 untuk menghindari pembagian dengan nol
t2 = np.linspace(2 + 1e-6, 6, 200)
y2 = 1 / (5 * t2 - 10) + 2
ax.plot(t2, y2, color='green', linewidth=2, label='f(t) = 1/(5t-10) + 2')
ax.plot(6, 2.05, 'o', markerfacecolor='white', markeredgecolor='green', markersize=8) # Titik kosong

# Menambahkan garis asimtot vertikal
ax.axvline(x=2, color='gray', linestyle='--', label='Asimtot Vertikal di t=2')

# 3. Segmen 3: Parabola [6, 9)
t3 = np.linspace(6, 9, 200)
y3 = (t3 - 7)**2 + 2
ax.plot(t3, y3, color='red', linewidth=2, label='f(t) = (t-7)² + 2')
ax.plot(6, 3, 'o', color='red', markersize=8) # Titik solid
ax.plot(9, 6, 'o', markerfacecolor='white', markeredgecolor='red', markersize=8) # Titik kosong

# 4. Segmen 4: Garis Horizontal [9, ∞)
t4 = np.linspace(9, 16, 100)
y4 = np.full_like(t4, 6)
ax.plot(t4, y4, color='purple', linewidth=2, label='f(t) = 6')
ax.plot(9, 6, 'o', color='purple', markersize=8) # Titik solid
ax.plot(11, 6, 'o', markerfacecolor='white', markeredgecolor='purple', markersize=8) # Lubang

# --- Pengaturan Tampilan Grafik ---
ax.set_title('Grafik Alokasi RAM (Versi Koreksi)', fontsize=16)
ax.set_xlabel('Waktu (t) dalam detik', fontsize=12)
ax.set_ylabel('Alokasi RAM f(t)', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.6)

# Batas sumbu disesuaikan untuk menunjukkan asimtot tanpa membuat grafik terlalu besar
ax.set_xlim(-1, 16)
ax.set_ylim(-5, 40)

# Menampilkan legenda dan plot
ax.legend()
plt.show()
# t=2
plt.scatter(2, f(2), color='blue', zorder=5)  
plt.scatter(2, np.nan, facecolors='none', edgecolors='orange', s=100, zorder=5)  # open placeholder (undefined)
# t=6
plt.scatter(6, f(6), color='orange', zorder=5)
plt.scatter(6, f(6-1e-6), facecolors='none', edgecolors='orange', s=100, zorder=5)  # open for hyperbola
# t=9
plt.scatter(9, f(9), color='red', zorder=5)  
plt.scatter(9, f(9+1e-6), facecolors='none', edgecolors='purple', s=100, zorder=5)  # open start rational

# Vertical lines
for tb in [2, 6, 9]:
    plt.axvline(tb, color='gray', linestyle='--', alpha=0.7)

plt.xlabel('t (detik)')
plt.ylabel('f(t) (Alokasi RAM)')
plt.title('Grafik Piecewise f(t) Proses Boot hingga Stabilitas')
plt.legend()
plt.grid(True)
plt.ylim(-1, 10)
plt.xlim(0, 15)
plt.show()
