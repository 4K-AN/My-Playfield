# 📚 Panduan Mencari dan Menambahkan Referenced Libraries di VS Code

## 🎯 Tujuan
Menambahkan MySQL JDBC Connector library ke project Java di VS Code agar aplikasi dapat terhubung dengan database.

---

## 🔍 Langkah-Langkah Detail

### **STEP 1: Buka Explorer di VS Code**

1. **Buka VS Code** dan pastikan project `DatabaseConnectionProject` sudah dibuka
2. **Klik icon Explorer** di sidebar sebelah kiri (icon file/folder)
   - Atau tekan: `Ctrl+Shift+E`

```
┌─────────────────────────────────┐
│  VS Code Sidebar (Kiri)         │
├─────────────────────────────────┤
│ 📁 EXPLORER (Klik di sini)      │
│ 🔎 SEARCH                       │
│ 📝 SOURCE CONTROL              │
│ ▶️  RUN AND DEBUG               │
│ 📦 EXTENSIONS                   │
└─────────────────────────────────┘
```

---

### **STEP 2: Cari Section "JAVA PROJECTS"**

Setelah mengklik Explorer, di bagian atas explorer panel akan terlihat beberapa section:

```
┌─────────────────────────────────────────┐
│  EXPLORER PANEL (Area Utama)            │
├─────────────────────────────────────────┤
│ 📂 DATABASE CONNECTION PROJECT          │
│   📁 src                                │
│   📁 bin                                │
│   📁 mysql-connector-j-9.3.0            │
│   📄 README.md                          │
│                                         │
│ ⬇️ Scroll ke bawah untuk menemukan:     │
│                                         │
│ JAVA PROJECTS                           │
│  └─ DatabaseConnectionProject           │
│      └─ Referenced Libraries            │
│                                         │
└─────────────────────────────────────────┘
```

**Jika tidak melihat "JAVA PROJECTS":**
- Scroll ke bawah di Explorer panel
- Atau klik arrow/chevron (▼) jika section sudah ada tapi tertutup

---

### **STEP 3: Expand "JAVA PROJECTS"**

Klik pada **"JAVA PROJECTS"** untuk membukanya:

```
JAVA PROJECTS  ▼  (Klik untuk expand/collapse)
  └─ DatabaseConnectionProject
```

Setelah di-expand, akan terlihat:

```
JAVA PROJECTS  ▼
  └─ DatabaseConnectionProject
     ├─ src/
     ├─ bin/
     └─ Referenced Libraries  ⭐ (Cari ini)
```

---

### **STEP 4: Temukan "Referenced Libraries"**

Di bawah `DatabaseConnectionProject`, cari menu **"Referenced Libraries"**:

```
JAVA PROJECTS
  └─ DatabaseConnectionProject
     ├─ src/
     │  └─ Main.java
     ├─ bin/
     │  └─ Main.class
     └─ Referenced Libraries  ⭐ KLIK DI SINI
        └─ (Awalnya kosong)
```

**Catatan:** Awalnya Referenced Libraries mungkin kosong atau hanya berisi standard library Java.

---

### **STEP 5: Klik Icon Tambah (+)**

Di samping "Referenced Libraries", akan terlihat **icon plus (+)**:

```
Referenced Libraries  ➕  ← Klik icon ini
```

Letaknya di sebelah kanan teks "Referenced Libraries".

**Jika tidak terlihat:**
- Hover mouse ke area "Referenced Libraries"
- Icon + akan muncul

```
Referenced Libraries  [Hover di sini] ➕
                     ↑ Icon akan muncul
```

---

### **STEP 6: Browse ke Folder mysql-connector-j-9.3.0**

Setelah klik icon +, akan muncul **dialog file browser**:

```
┌──────────────────────────────────────────────┐
│  Select Library JAR                          │
├──────────────────────────────────────────────┤
│  Pilih lokasi folder:                        │
│  📂 DatabaseConnectionProject (current)      │
│     📂 src                                   │
│     📂 bin                                   │
│     📂 mysql-connector-j-9.3.0  ⭐ MASUK KE│
│     📄 README.md                             │
│                                              │
└──────────────────────────────────────────────┘
```

**Langkah-langkahnya:**
1. Cari folder: `mysql-connector-j-9.3.0` di folder project
2. **Double-click** folder tersebut untuk masuk ke dalamnya

---

### **STEP 7: Pilih File JAR**

Setelah masuk ke folder `mysql-connector-j-9.3.0`, akan terlihat:

```
┌──────────────────────────────────────────────┐
│  Select Library JAR                          │
├──────────────────────────────────────────────┤
│  Folder: mysql-connector-j-9.3.0/            │
│                                              │
│  📄 mysql-connector-j-9.3.0.jar ⭐ PILIH   │
│  📄 LICENSE                                  │
│  📄 README                                   │
│  📄 CHANGELOG                                │
│                                              │
│  [Cancel]  [Select] (tombol di bawah)       │
│                                              │
└──────────────────────────────────────────────┘
```

**Pilih file:**
- Klik file: `mysql-connector-j-9.3.0.jar` (file dengan extension .jar)
- Klik tombol **"Select"** atau **"Open"**

---

### **STEP 8: Verifikasi Library Sudah Ditambahkan**

Setelah memilih file, kembali ke Explorer dan lihat:

```
JAVA PROJECTS
  └─ DatabaseConnectionProject
     ├─ src/
     ├─ bin/
     └─ Referenced Libraries  ✓
        └─ mysql-connector-j-9.3.0.jar ✅ BERHASIL!
```

**Library sudah berhasil ditambahkan jika:**
- ✅ File `mysql-connector-j-9.3.0.jar` muncul di bawah Referenced Libraries
- ✅ Tidak ada error message

---

## 🎨 Visual Diagram Lengkap

### Dari Awal hingga Selesai:

```
1. Buka Explorer (Ctrl+Shift+E)
        ↓
2. Scroll dan cari "JAVA PROJECTS"
        ↓
3. Expand DatabaseConnectionProject
        ↓
4. Temukan "Referenced Libraries"
        ↓
5. Klik icon + (Plus) di samping Referenced Libraries
        ↓
6. Browse ke folder: mysql-connector-j-9.3.0
        ↓
7. Pilih file: mysql-connector-j-9.3.0.jar
        ↓
8. Klik Select/Open
        ↓
✅ SELESAI! Library sudah ditambahkan
```

---

## ❓ FAQ & Troubleshooting

### P: Tidak bisa menemukan "JAVA PROJECTS"?
**J:** 
- Pastikan Extension Pack for Java sudah diinstall
- Restart VS Code
- Scroll ke bawah di Explorer panel
- Pastikan folder project sudah dibuka

### P: Icon + tidak muncul di Referenced Libraries?
**J:**
- Hover mouse ke area "Referenced Libraries"
- Icon akan muncul saat mouse hover
- Pastikan di-extend dulu (chevron/arrow dibuka)

### P: File .jar tidak terlihat di folder?
**J:**
- Pastikan file sudah diekstrak dari .zip
- Verifikasi file ada di: `DatabaseConnectionProject/mysql-connector-j-9.3.0/mysql-connector-j-9.3.0.jar`
- Gunakan Windows Explorer untuk cek file

### P: Setelah ditambahkan, masih ada error?
**J:**
- Restart VS Code
- Rebuild project: Ctrl+Shift+B
- Clear dan compile ulang

---

## 📸 Screenshot Reference

**Lokasi Explorer:**
```
┌─────────────┐
│ 📁 Explorer │ ← Klik ikon ini (atau Ctrl+Shift+E)
│ 🔍          │
│ 📝          │
└─────────────┘
```

**Lokasi JAVA PROJECTS:**
```
EXPLORER
├─ DATABASE CONNECTION PROJECT
├─ JAVA PROJECTS ← Scroll ke bawah
│  └─ DatabaseConnectionProject
│     └─ Referenced Libraries ← Klik + di sini
```

---

## ✅ Checklist Penyelesaian

- [ ] Buka Explorer (Ctrl+Shift+E)
- [ ] Cari dan expand "JAVA PROJECTS"
- [ ] Temukan "Referenced Libraries"
- [ ] Klik icon + di samping Referenced Libraries
- [ ] Browse ke folder mysql-connector-j-9.3.0
- [ ] Pilih file mysql-connector-j-9.3.0.jar
- [ ] Klik Select/Open
- [ ] Verifikasi library muncul di Referenced Libraries
- [ ] Restart VS Code jika perlu
- [ ] Aplikasi siap untuk compile & run dengan JDBC

---

## 🚀 Setelah Library Ditambahkan

Sekarang Anda bisa:
1. ✅ Compile aplikasi Java dengan JDBC support
2. ✅ Run aplikasi dan connect ke MySQL database
3. ✅ Execute query SQL dari Java

Jalankan aplikasi dengan: `Ctrl+F5`

---

**Modul 12: Pemrograman SQL - Pengembangan Aplikasi Basis Data Menggunakan Java**

Jika masih ada pertanyaan, lihat file `GETTING_STARTED.md` untuk panduan lengkap!
