import numpy as np
import matplotlib.pyplot as plt

# Definisi fungsi piecewise yang benar
def f(t):
    if isinstance(t, np.ndarray):
        result = np.zeros_like(t)
        mask1 = (t >= 0) & (t < 2)
        mask2 = (t >= 2) & (t < 6)
        mask3 = (t >= 6) & (t < 9)
        mask4 = (t >= 9) & (t != 11)  # Exclude t=11 karena tidak terdefinisi
        
        result[mask1] = 2 * t[mask1]
        result[mask2] = 1 / (5 * t[mask2] - 10) + 2
        result[mask3] = (t[mask3] - 7)**2 + 2
        result[mask4] = (6 * t[mask4] - 66) / (t[mask4] - 11)
        
        return result
    else:
        if 0 <= t < 2:
            return 2 * t
        elif 2 <= t < 6:
            return 1 / (5 * t - 10) + 2
        elif 6 <= t < 9:
            return (t - 7)**2 + 2
        elif t >= 9 and t != 11:
            return (6 * t - 66) / (t - 11)
        else:
            return np.nan

# Membuat array waktu untuk setiap interval
t1 = np.linspace(0, 1.99, 100)      # [0, 2)
t2 = np.linspace(2.01, 5.99, 200)   # (2, 6) - hindari t=2 untuk menghindari pembagian nol
t3 = np.linspace(6, 8.99, 150)      # [6, 9)
t4a = np.linspace(9, 10.99, 100)    # [9, 11) - sebelum asimtot
t4b = np.linspace(11.01, 15, 200)   # (11, 15] - setelah asimtot

# Menghitung nilai fungsi untuk setiap interval
# Menghitung nilai fungsi untuk setiap intervald
f1 = 2 * t1
f2 = 1 / (5 * t2 - 10) + 2
f3 = (t3 - 7)**2 + 2
f4a = (6 * t4a - 66) / (t4a - 11)
f4b = (6 * t4b - 66) / (t4b - 11)

# Membuat plot
plt.figure(figsize=(16, 12))

# Plot setiap bagian dengan warna berbeda
plt.plot(t1, f1, 'b-', linewidth=3, label='f(t) = 2t (Boot time: 0 ≤ t < 2)')
plt.plot(t2, f2, 'r-', linewidth=3, label='f(t) = 1/(5t-10) + 2 (After boot: 2 ≤ t < 6)')
plt.plot(t3, f3, 'g-', linewidth=3, label='f(t) = (t-7)² + 2 (Video app: 6 ≤ t < 9)')
plt.plot(t4a, f4a, 'm-', linewidth=3, label='f(t) = (6t-66)/(t-11) (Stabilization: t ≥ 9)')
plt.plot(t4b, f4b, 'm-', linewidth=3)

# Menghitung nilai di titik-titik kritis (hindari pembagian nol)
values = {}
values[2] = {
    'left': 2 * 2,  # f(2⁻) = 2(2) = 4
    'right': 'undefined',  # f(2⁺) = 1/0 + 2 → tidak terdefinisi (asimtot vertikal)
    'value': 'undefined'   # f(2) tidak terdefinisi
}

values[6] = {
    'left': 1 / (5 * 6 - 10) + 2,   # f(6⁻) = 1/20 + 2 = 2.05
    'right': (6 - 7)**2 + 2,        # f(6⁺) = 1 + 2 = 3
    'value': (6 - 7)**2 + 2         # f(6) = 3
}

values[9] = {
    'left': (9 - 7)**2 + 2,         # f(9⁻) = 4 + 2 = 6
    'right': (6 * 9 - 66) / (9 - 11), # f(9⁺) = -12/(-2) = 6
    'value': (6 * 9 - 66) / (9 - 11)  # f(9) = 6
}

values[11] = {
    'left': 'undefined',
    'right': 'undefined',
    'value': 'undefined',
    'limit': 6
}

# Plot titik-titik kritis yang terdefinisi
plt.plot(6, values[6]['value'], 'ko', markersize=10)
plt.plot(9, values[9]['value'], 'ko', markersize=10)

# Untuk t=6: gambar diskontinuitas
plt.plot(6, values[6]['left'], 'ro', markersize=8, fillstyle='none', markeredgewidth=2)
plt.plot(6, values[6]['right'], 'go', markersize=8)

# Asimtot vertikal di t = 2 dan t = 11
plt.axvline(x=2, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Asimtot vertikal: t = 2')
plt.axvline(x=11, color='orange', linestyle='--', linewidth=2, alpha=0.8, label='Asimtot vertikal: t = 11')

# Asimtot horizontal di y = 6 untuk interval terakhir
plt.axhline(y=6, color='purple', linestyle=':', linewidth=2, alpha=0.8, label='Asimtot horizontal: y = 6')

# Garis vertikal untuk titik transisi
for t in [2, 6, 9]:
    plt.axvline(x=t, color='gray', linestyle='--', alpha=0.5)

# Mengatur sumbu dan label
plt.xlabel('Waktu t (detik)', fontsize=14, fontweight='bold')
plt.ylabel('RAM Allocation f(t)', fontsize=14, fontweight='bold')
plt.title('Grafik Fungsi RAM Allocation vs Waktu\n(Piecewise Function - Koreksi)', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11, loc='upper right')

# Mengatur batas sumbu untuk visualisasi yang lebih baik
plt.xlim(-0.5, 15)
plt.ylim(-10, 15)

# Menambahkan anotasi untuk nilai di titik kritis
annotations = [
    (2, 8, 't=2: Asimtot vertikal\nf(2) tidak terdefinisi', 'red'),
    (6, -5, f't=6: f(6⁻)=2.05, f(6⁺)=3, f(6)=3\nTidak kontinu', 'red'),
    (9, 8, f't=9: f(9⁻)=6, f(9⁺)=6, f(9)=6\nKontinu', 'green'),
    (11, -8, f't=11: Asimtot vertikal\nf(11) tidak terdefinisi, limit=6', 'orange')
]

for t, y, text, color in annotations:
    plt.annotate(text, xy=(t, y), xytext=(t+1.5, y+1),
                bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.3),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5),
                fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

# Analisis kontinuitas secara detail
print("="*80)
print("ANALISIS KONTINUITAS FUNGSI f(t) - KOREKSI")
print("="*80)

def analyze_continuity_corrected(t_point):
    print(f"\n{'='*60}")
    print(f"KONTINUITAS DI t = {t_point}")
    print(f"{'='*60}")
    
    if t_point == 2:
        left_limit = 2 * 2
        print(f"Limit kiri:  lim(t→2⁻) f(t) = lim(t→2⁻) 2t = 2(2) = {left_limit}")
        print(f"Limit kanan: lim(t→2⁺) f(t) = lim(t→2⁺) [1/(5t-10) + 2]")
        print(f"                                = lim(t→2⁺) [1/(5(2)-10) + 2]")
        print(f"                                = lim(t→2⁺) [1/0 + 2] = ±∞")
        print(f"Nilai fungsi: f(2) = 1/(5×2-10) + 2 = 1/0 + 2 → TIDAK TERDEFINISI")
        print(f"❌ TIDAK KONTINU: f(2) tidak terdefinisi")
        print(f"   Jenis: Infinite Discontinuity (Asimtot Vertikal)")
    
    elif t_point == 6:
        left_limit = 1 / (5 * 6 - 10) + 2
        right_limit = (6 - 7)**2 + 2
        func_value = (6 - 7)**2 + 2
        
        print(f"Limit kiri:  lim(t→6⁻) f(t) = lim(t→6⁻) [1/(5t-10) + 2]")
        print(f"                                = 1/(5×6-10) + 2 = 1/20 + 2 = {left_limit}")
        print(f"Limit kanan: lim(t→6⁺) f(t) = lim(t→6⁺) [(t-7)² + 2]")
        print(f"                                = (6-7)² + 2 = 1 + 2 = {right_limit}")
        print(f"Nilai fungsi: f(6) = (6-7)² + 2 = {func_value}")
        
        if abs(left_limit - right_limit) < 1e-10:
            print(f"✅ KONTINU: Semua nilai sama = {func_value}")
        else:
            print(f"❌ TIDAK KONTINU: {left_limit} ≠ {right_limit}")
            print(f"   Jenis: Jump Discontinuity (lompatan sebesar {abs(right_limit - left_limit):.3f})")
    
    elif t_point == 9:
        left_limit = (9 - 7)**2 + 2
        right_limit = (6 * 9 - 66) / (9 - 11)
        func_value = (6 * 9 - 66) / (9 - 11)
        
        print(f"Limit kiri:  lim(t→9⁻) f(t) = lim(t→9⁻) [(t-7)² + 2]")
        print(f"                                = (9-7)² + 2 = 4 + 2 = {left_limit}")
        print(f"Limit kanan: lim(t→9⁺) f(t) = lim(t→9⁺) (6t-66)/(t-11)")
        print(f"                                = (6×9-66)/(9-11) = (54-66)/(-2)")
        print(f"                                = -12/(-2) = {right_limit}")
        print(f"Nilai fungsi: f(9) = (6×9-66)/(9-11) = {func_value}")
        
        if abs(left_limit - right_limit) < 1e-10 and abs(left_limit - func_value) < 1e-10:
            print(f"✅ KONTINU: Semua nilai sama = {func_value}")
        else:
            print(f"❌ TIDAK KONTINU")
    
    elif t_point == 11:
        print(f"Nilai fungsi: f(11) = (6×11-66)/(11-11) = (66-66)/0 = 0/0")
        print(f"                     → TIDAK TERDEFINISI")
        print(f"Limit: lim(t→11) (6t-66)/(t-11)")
        print(f"     = lim(t→11) 6(t-11)/(t-11)  [faktoring pembilang]")
        print(f"     = lim(t→11) 6 = 6")
        print(f"❌ TIDAK KONTINU: f(11) tidak terdefinisi")
        print(f"   Jenis: Removable Discontinuity (dapat diperbaiki dengan f(11) = 6)")

# Analisis setiap titik
for t in [2, 6, 9, 11]:
    analyze_continuity_corrected(t)

print(f"\n{'='*80}")
print("RINGKASAN KONTINUITAS - KOREKSI")
print(f"{'='*80}")
print("t = 2:  ❌ TIDAK KONTINU (Infinite Discontinuity - Asimtot Vertikal)")
print("t = 6:  ❌ TIDAK KONTINU (Jump Discontinuity)")  
print("t = 9:  ✅ KONTINU")
print("t = 11: ❌ TIDAK KONTINU (Removable Discontinuity)")
print(f"{'='*80}")

# Tabel nilai fungsi untuk referensi
print(f"\n{'='*80}")
print("TABEL NILAI FUNGSI DI TITIK KRITIS")
print(f"{'='*80}")
print("t    | f(t⁻)        | f(t⁺)   | f(t)      | Status")
print("-" * 55)
print("2    | 4            | ±∞      | undefined | Tidak Kontinu")
print("6    | 2.05         | 3       | 3         | Tidak Kontinu") 
print("9    | 6            | 6       | 6         | Kontinu")
print("11   | -            | -       | undefined | Tidak Kontinu")
print(f"{'='*80}")