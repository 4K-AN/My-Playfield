# Setup Guide - Pengembangan Aplikasi Basis Data Menggunakan Java

## Modul 12: Pemrograman SQL

Panduan lengkap untuk mengintegrasikan aplikasi Java dengan MySQL Database.

---

## 1. Penyiapan Lingkungan Kerja

### 1.1 MySQL Server dan MySQL Workbench
- Download dari: https://dev.mysql.com/downloads/installer/
- Pilih installer yang sesuai dengan sistem operasi
- Install dan catat username, password, serta port yang digunakan (default: 3306)

### 1.2 MySQL JDBC Connector
- Download dari: https://dev.mysql.com/downloads/connector/j/
- Pilih "Platform Independent" untuk kompatibilitas maksimal
- Download file: `mysql-connector-j-9.3.0.zip` (atau versi terbaru)
- Ekstrak file di dalam folder project

### 1.3 Database Akademik
Pastikan database `akademik` sudah dibuat di MySQL Server dengan tabel `mahasiswa` dan kolom-kolom berikut:

```sql
CREATE TABLE mahasiswa (
    NIM VARCHAR(20) PRIMARY KEY,
    ID_Seleksi_Masuk INT,
    ID_Program_Studi INT,
    nama VARCHAR(100),
    angkatan INT,
    tgl_lahir DATE,
    kota_lahir VARCHAR(50),
    jenis_kelamin CHAR(1),
    ipk DECIMAL(3,2)
);
```

---

## 2. Penyiapan Visual Studio Code

### 2.1 Install Extension Java
1. Buka VS Code
2. Klik menu Extensions (Ctrl+Shift+X)
3. Cari "Extension Pack for Java" oleh Microsoft
4. Install extension tersebut

### 2.2 Struktur Project
Project sudah tersedia dengan struktur:
```
DatabaseConnectionProject/
├── src/
│   └── Main.java          (File aplikasi utama)
├── bin/                   (Folder output binary)
├── mysql-connector-j-9.3.0/
│   └── mysql-connector-j-9.3.0.jar  (JDBC Library)
└── README.md
```

### 2.3 Konfigurasi JDBC Library
1. Buka Explorer di VS Code
2. Cari section "JAVA PROJECTS"
3. Expand "Referenced Libraries"
4. Klik icon + (Add)
5. Navigate ke folder `mysql-connector-j-9.3.0`
6. Pilih file `mysql-connector-j-9.3.0.jar`
7. Verifikasi library sudah terpasang

---

## 3. Konfigurasi Koneksi Database

Buka file `Main.java` dan sesuaikan parameter koneksi:

```java
private static final String URL = "jdbc:mysql://localhost:3306/akademik";
private static final String USERNAME = "root";
private static final String PASSWORD = "root";
```

**Catatan:**
- Ganti `3306` jika port MySQL Anda berbeda
- Ganti `root` dengan username MySQL Anda
- Ganti password sesuai konfigurasi MySQL

---

## 4. Menjalankan Aplikasi

### 4.1 Test Koneksi
1. Di `Main.java`, pastikan hanya `testConnection()` yang di-uncomment
2. Jalankan dengan: `Ctrl+F5` atau klik Run
3. Jika berhasil, akan muncul pesan: "✓ Koneksi Berhasil!"

### 4.2 Menampilkan Data Mahasiswa
1. Comment method `testConnection()`
2. Uncomment method `displayMahasiswaData()`
3. Jalankan aplikasi
4. Data mahasiswa akan ditampilkan di terminal

### 4.3 Menginput Data Mahasiswa
1. Comment method sebelumnya
2. Uncomment method `insertMahasiswaData()`
3. Jalankan aplikasi
4. Data baru akan ditambahkan ke database

### 4.4 Update Data Mahasiswa
1. Uncomment method `updateMahasiswaData()`
2. Jalankan aplikasi
3. Data mahasiswa akan diupdate

### 4.5 Hapus Data Mahasiswa
1. Uncomment method `deleteMahasiswaData()`
2. Jalankan aplikasi
3. Data mahasiswa akan dihapus

---

## 5. Troubleshooting

### Koneksi Gagal
**Error:** "Koneksi Gagal"

**Solusi:**
- Pastikan MySQL Server sedang berjalan
- Verifikasi database `akademik` sudah dibuat
- Periksa username, password, dan port di Main.java
- Pastikan JDBC Library sudah ditambahkan di Referenced Libraries

### JDBC Library Tidak Ditemukan
**Error:** "ClassNotFoundException: com.mysql.cj.jdbc.Driver"

**Solusi:**
- Download ulang `mysql-connector-j` dari link resmi
- Ekstrak dan tambahkan `.jar` file ke Referenced Libraries
- Restart VS Code jika perlu

### Data Tidak Ditampilkan
**Error:** "Tidak ada data mahasiswa di database"

**Solusi:**
- Pastikan tabel `mahasiswa` memiliki data
- Jalankan query INSERT terlebih dahulu
- Verifikasi di MySQL Workbench bahwa data sudah ada

---

## 6. Referensi Kode

### Class dan Method Utama
- `testConnection()` - Test koneksi ke database
- `displayMahasiswaData()` - Query dan tampilkan data
- `insertMahasiswaData()` - Tambah data baru
- `updateMahasiswaData()` - Ubah data existing
- `deleteMahasiswaData()` - Hapus data

### Import yang Digunakan
```java
import java.sql.Connection;      // Koneksi database
import java.sql.DriverManager;   // Manager driver JDBC
import java.sql.ResultSet;       // Hasil query
import java.sql.SQLException;    // Error handling
import java.sql.Statement;       // Eksekusi query
```

---

## 7. Build dan Run dari Terminal

### Compile
```bash
javac -cp "mysql-connector-j-9.3.0/mysql-connector-j-9.3.0.jar" -d bin src/Main.java
```

### Run
```bash
java -cp "bin;mysql-connector-j-9.3.0/mysql-connector-j-9.3.0.jar" Main
```

---

**Dibuat berdasarkan: Modul 12 Pemrograman SQL - Pengembangan Aplikasi Basis Data Menggunakan Java**
