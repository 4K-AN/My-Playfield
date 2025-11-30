# 🚀 Getting Started Guide

## DatabaseConnectionProject - Java & MySQL Integration

Panduan singkat untuk memulai aplikasi Java yang terkoneksi dengan MySQL Database.

---

## ✅ Checklist Penyiapan

### Step 1: Instal MySQL & MySQL Workbench
- [ ] Download dari: https://dev.mysql.com/downloads/installer/
- [ ] Install MySQL Server
- [ ] Catat: `username`, `password`, dan `port` (default: 3306)
- [ ] Buka MySQL Workbench untuk verifikasi

### Step 2: Download & Setup JDBC
- [ ] Download MySQL Connector/J dari: https://dev.mysql.com/downloads/connector/j/
- [ ] Pilih "Platform Independent"
- [ ] Download: `mysql-connector-j-9.3.0.zip`
- [ ] Ekstrak ke folder project
- [ ] Pastikan struktur: `mysql-connector-j-9.3.0/mysql-connector-j-9.3.0.jar`

### Step 3: Setup Database
- [ ] Buka MySQL Workbench
- [ ] Copy & jalankan SQL dari `setup_database.sql`
- [ ] Verifikasi tabel `mahasiswa` sudah dibuat
- [ ] Verifikasi data sample sudah masuk

### Step 4: Konfigurasi VS Code
- [ ] Buka Visual Studio Code
- [ ] Install Extension: "Extension Pack for Java"
- [ ] Buka project folder (DatabaseConnectionProject)
- [ ] Open Explorer → JAVA PROJECTS → Referenced Libraries
- [ ] Klik + → pilih `mysql-connector-j-9.3.0.jar`
- [ ] Restart VS Code jika perlu

### Step 5: Update Credentials
- [ ] Buka file: `src/Main.java`
- [ ] Edit line 10-12 sesuai MySQL Anda:
  ```java
  private static final String URL = "jdbc:mysql://localhost:3306/akademik";
  private static final String USERNAME = "root";           // Sesuaikan
  private static final String PASSWORD = "root";           // Sesuaikan
  ```

---

## 🎯 Quick Test

### Test Koneksi (Recommended First)

1. Buka `src/Main.java`
2. Di method `main()`, pastikan hanya ini yang **tidak** ter-comment:
   ```java
   testConnection();
   ```
3. Semua method lain harus di-comment dengan `//`

4. Jalankan dengan: `Ctrl+F5`

**Hasil Sukses:**
```
✓ Koneksi Berhasil!
Database: akademik
Host: localhost
Port: 3306
```

**Hasil Error:**
- Pastikan MySQL running
- Cek username, password, port
- Cek database `akademik` sudah ada

---

## 📊 Operasi Database

Setelah koneksi berhasil, uncomment salah satu operasi:

### 1️⃣ Tampilkan Data
```java
// testConnection();                    // Comment ini
displayMahasiswaData();                 // Uncomment ini
// insertMahasiswaData();
// updateMahasiswaData();
// deleteMahasiswaData();
```
**Hasil:** Semua data mahasiswa akan ditampilkan

### 2️⃣ Tambah Data
```java
insertMahasiswaData();
```
**Hasil:** Data baru mahasiswa akan ditambahkan

### 3️⃣ Update Data
```java
updateMahasiswaData();
```
**Hasil:** Data mahasiswa akan diubah

### 4️⃣ Hapus Data
```java
deleteMahasiswaData();
```
**Hasil:** Data mahasiswa akan dihapus

---

## 🔌 Troubleshooting

### ❌ Error: "Koneksi Gagal"
**Penyebab Umum:**
1. MySQL Server tidak running
2. Database `akademik` belum dibuat
3. Username/password salah
4. Port tidak sesuai

**Solusi:**
```bash
# Windows - Cek MySQL Service
sc query MySQL80

# Buka Services dan start MySQL jika needed
# Atau restart dari MySQL Workbench
```

### ❌ Error: "ClassNotFoundException"
**Penyebab:** JDBC Library tidak terpasang

**Solusi:**
1. Buka VS Code Explorer
2. Cari: JAVA PROJECTS → Referenced Libraries
3. Klik + dan pilih `mysql-connector-j-9.3.0.jar`

### ❌ Error: "Table doesn't exist"
**Penyebab:** Tabel `mahasiswa` belum dibuat

**Solusi:**
1. Buka MySQL Workbench
2. Jalankan SQL dari `setup_database.sql`
3. Verify via query: `SELECT * FROM mahasiswa;`

### ❌ Data Tidak Muncul
**Penyebab:** Database kosong

**Solusi:**
1. Jalankan `insertMahasiswaData()` untuk menambah data
2. Atau import data dari `setup_database.sql`
3. Verifikasi di MySQL Workbench

---

## 📁 File Penting

| File | Fungsi |
|------|--------|
| `src/Main.java` | Aplikasi utama dengan semua method |
| `setup_database.sql` | Script setup database MySQL |
| `SETUP_GUIDE.md` | Panduan setup lengkap |
| `README.md` | Project overview |
| `.vscode/tasks.json` | Build & run tasks |

---

## 💡 Tips & Tricks

### Build Dari Terminal
```bash
javac -cp "mysql-connector-j-9.3.0/mysql-connector-j-9.3.0.jar" -d bin src/Main.java
```

### Run Dari Terminal
```bash
java -cp "bin;mysql-connector-j-9.3.0/mysql-connector-j-9.3.0.jar" Main
```

### Clear Compiled Files
```bash
rmdir /s /q bin
mkdir bin
```

### View MySQL Data
```bash
# Di MySQL Workbench atau Command Line
USE akademik;
SELECT * FROM mahasiswa;
```

---

## 🎓 Learning Path

1. ✅ Test koneksi database
2. ✅ Tampilkan data dari database
3. ✅ Insert data baru
4. ✅ Update data existing
5. ✅ Delete data

Setiap step sudah disediakan method-nya di `Main.java`

---

## 📚 Referensi

- **MySQL JDBC Documentation:** https://dev.mysql.com/doc/connector-j/
- **Java SQL Tutorial:** https://docs.oracle.com/javase/tutorial/jdbc/
- **MySQL Download:** https://dev.mysql.com/downloads/
- **VS Code Java:** https://code.visualstudio.com/docs/languages/java

---

## 🤝 Need Help?

1. **Connection Issues:** Lihat SETUP_GUIDE.md section 3
2. **SQL Errors:** Verify database di MySQL Workbench
3. **Java Errors:** Rebuild project (Ctrl+Shift+B)
4. **Library Issues:** Re-add JDBC di Referenced Libraries

---

**Happy Coding! 🎉**

Modul 12: Pemrograman SQL - Pengembangan Aplikasi Basis Data Menggunakan Java
