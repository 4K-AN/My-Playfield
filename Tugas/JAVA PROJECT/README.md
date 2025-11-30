# Database Akademik - Java Application

Aplikasi Java untuk pengintegrasian dengan MySQL Database sesuai dengan **Modul 12: Pemrograman SQL - Pengembangan Aplikasi Basis Data Menggunakan Java**.

## 📋 Overview

Aplikasi ini mendemonstrasikan:
- ✅ Koneksi Java dengan MySQL Database
- ✅ Query SQL (SELECT, INSERT, UPDATE, DELETE)
- ✅ Manipulasi data dari aplikasi Java
- ✅ Error handling dan connection management

## 📁 Struktur Project

```
DatabaseConnectionProject/
├── src/
│   └── Main.java                    # File aplikasi utama dengan semua fungsi
├── bin/                             # Output directory untuk compiled files
├── mysql-connector-j-9.3.0/
│   └── mysql-connector-j-9.3.0.jar  # JDBC Driver
├── README.md                        # File ini
└── SETUP_GUIDE.md                   # Panduan setup lengkap
```

## 🚀 Quick Start

### 1. Install Prerequisites
- MySQL Server & MySQL Workbench
- Visual Studio Code
- Extension Pack for Java (via VS Code Extensions)
- MySQL JDBC Connector 9.3.0

### 2. Setup Database
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

### 3. Configure Connection
Edit `src/Main.java` line 10-12:
```java
private static final String URL = "jdbc:mysql://localhost:3306/akademik";
private static final String USERNAME = "root";
private static final String PASSWORD = "root";
```

### 4. Add JDBC Library
1. Open Explorer → JAVA PROJECTS → Referenced Libraries
2. Click + icon
3. Navigate to `mysql-connector-j-9.3.0` folder
4. Select `.jar` file

### 5. Run Application
Press `Ctrl+F5` or click Run button

## 📝 Available Functions

### 1. Test Koneksi
```java
testConnection();  // Uncomment untuk test koneksi
```

### 2. Menampilkan Data
```java
displayMahasiswaData();  // Query dan tampilkan semua mahasiswa
```

### 3. Menginput Data
```java
insertMahasiswaData();  // Tambah data mahasiswa baru
```

### 4. Update Data
```java
updateMahasiswaData();  // Ubah data mahasiswa
```

### 5. Hapus Data
```java
deleteMahasiswaData();  // Hapus data mahasiswa
```

## 🔧 Terminal Commands

### Compile
```bash
javac -cp "mysql-connector-j-9.3.0/mysql-connector-j-9.3.0.jar" -d bin src/Main.java
```

### Run
```bash
java -cp "bin;mysql-connector-j-9.3.0/mysql-connector-j-9.3.0.jar" Main
```

## 📚 Key Concepts

### Connection
```java
Connection conn = DriverManager.getConnection(url, username, password);
```

### Statement & Query
```java
Statement stmt = conn.createStatement();
String query = "SELECT * FROM mahasiswa";
```

### ResultSet
```java
ResultSet rs = stmt.executeQuery(query);
while (rs.next()) {
    String nim = rs.getString("NIM");
    String nama = rs.getString("nama");
}
```

### Execute Update
```java
String insertQuery = "INSERT INTO mahasiswa ...";
int rowsAffected = stmt.executeUpdate(insertQuery);
```

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Connection Gagal | Pastikan MySQL running, database exist, credentials benar |
| JDBC Not Found | Download dari https://dev.mysql.com/downloads/connector/j/ |
| Data Tidak Ditampilkan | Verifikasi data ada di database via MySQL Workbench |
| Compilation Error | Pastikan JDBC library sudah ditambahkan di Referenced Libraries |

## 📖 Learn More

- Lihat `SETUP_GUIDE.md` untuk panduan setup lengkap
- MySQL Documentation: https://dev.mysql.com/doc/
- Java JDBC: https://docs.oracle.com/javase/tutorial/jdbc/
- Download JDBC: https://dev.mysql.com/downloads/connector/j/

## 📝 Notes

- Sesuaikan port MySQL jika tidak menggunakan default (3306)
- Gunakan MySQL Workbench untuk verifikasi data di database
- Comment/uncomment method di `main()` untuk menjalankan operasi yang berbeda

---

**Modul 12: Pemrograman SQL**  
**Pengembangan Aplikasi Basis Data Menggunakan Java**

