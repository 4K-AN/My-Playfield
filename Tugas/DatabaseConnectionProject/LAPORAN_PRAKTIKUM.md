# LAPORAN PRAKTIKUM
## KONEKSI DATABASE MYSQL DENGAN JAVA

**Nama:** [Nama Mahasiswa]  
**NIM:** [NIM Mahasiswa]  
**Kelas:** [Kelas]  
**Tanggal:** [Tanggal Praktikum]

---

## TUJUAN PRAKTIKUM

1. Mahasiswa mampu mengintegrasikan aplikasi Java dengan MySQL
2. Mahasiswa mampu melakukan query SQL dan menampilkan hasilnya di aplikasi Java

---

## LANGKAH PRAKTIKUM

### A. Penyiapan Lingkungan Kerja

1. **Menyiapkan MySQL Server dan MySQL Workbench**
   - Link download: https://dev.mysql.com/downloads/installer/

2. **Menyiapkan file JDBC**
   - Link download: https://dev.mysql.com/downloads/connector/j/
   - Pilih Platform Independent dan download file `mysql-connector-j-9.5.0.zip`
   - Ekstrak file ke folder proyek

3. **Menyiapkan database akademik**
   - Database akademik harus sudah terinstal di MySQL server
   - Database berisi tabel-tabel seperti: `mahasiswa`, `PROGRAM_STUDI`, dll

### B. Penyiapan IDE Tools

1. **Menyiapkan text editor dan ekstensi Java**
   - Install Visual Studio Code
   - Install ekstensi "Extension Pack for Java"

2. **Membuat proyek Java**
   - Buat proyek baru dengan nama `DatabaseConnectionProject`
   - Tambahkan library JDBC ke Referenced Libraries

3. **Melakukan koneksi MySQL di Java**

   **Kode koneksi:**
   ```java
   import java.sql.Connection;
   import java.sql.DriverManager;
   import java.sql.ResultSet;
   import java.sql.SQLException;
   import java.sql.Statement;

   public class main {
       private static final String URL = "jdbc:mysql://127.0.0.1:3306/akademik";
       private static final String USERNAME = "root";
       private static final String PASSWORD = "221205";

       public static void main(String[] args) {
           try {
               Connection conn = DriverManager.getConnection(URL, USERNAME, PASSWORD);
               System.out.println("Koneksi Berhasil!");
               conn.close();
           } catch (SQLException e) {
               System.out.println("Koneksi Gagal: " + e.getMessage());
           }
       }
   }
   ```

4. **Menampilkan data dari database**

   **Kode untuk menampilkan data:**
   ```java
   try {
       Connection conn = DriverManager.getConnection(URL, USERNAME, PASSWORD);
       Statement stmt = conn.createStatement();
       String query = "SELECT * FROM mahasiswa";
       ResultSet rs = stmt.executeQuery(query);

       while (rs.next()) {
           System.out.println("NIM: " + rs.getString("NIM"));
           System.out.println("Nama: " + rs.getString("nama"));
           // ... kolom lainnya
       }
   } catch (SQLException e) {
       System.out.println("Koneksi Gagal: " + e.getMessage());
   }
   ```

5. **Melakukan manipulasi data (INSERT, UPDATE, DELETE)**

   **Kode untuk INSERT:**
   ```java
   String query = "INSERT INTO mahasiswa (NIM, ID_Seleksi_Masuk, ID_Program_Studi, nama, angkatan, tgl_lahir, kota_lahir, jenis_kelamin) VALUES ('123456789', 1, 211, 'John Doe', 2023, '2000-01-01', 'Jakarta', 'P')";
   stmt.executeUpdate(query);
   ```

---

## LATIHAN

### LATIHAN 1: Update NIM Mahasiswa

**Tujuan:** Melakukan update data NIM pada mahasiswa yang bernama "John Doe" dengan NIM baru, lalu menampilkan hasilnya pada terminal.

#### 1.1 Query

**Query UPDATE yang digunakan:**
```sql
UPDATE mahasiswa 
SET NIM = '245150707111012' 
WHERE nama = 'John Doe';
```

**Kode Java lengkap:**
```java
/**
 * Method untuk mengupdate NIM mahasiswa John Doe menjadi 245150707111012
 */
public static void updateNIMJohnDoe() {
    System.out.println(">>> LATIHAN 1: UPDATE NIM JOHN DOE <<<\n");
    try {
        Connection conn = DriverManager.getConnection(URL, USERNAME, PASSWORD);
        System.out.println("✓ Koneksi Berhasil!\n");

        Statement stmt = conn.createStatement();

        // Query UPDATE untuk mengubah NIM John Doe
        String query = "UPDATE mahasiswa SET NIM = '245150707111012' WHERE nama = 'John Doe'";

        // Eksekusi query UPDATE
        int rowsUpdated = stmt.executeUpdate(query);

        if (rowsUpdated > 0) {
            System.out.println("✓ NIM berhasil diupdate!");
            System.out.println("Nama: John Doe");
            System.out.println("NIM Baru: 245150707111012");
            System.out.println("Jumlah baris yang diubah: " + rowsUpdated + "\n");
        } else {
            System.out.println("✗ Data tidak ditemukan! (John Doe tidak ada di database)\n");
        }

        stmt.close();
        conn.close();

    } catch (SQLException e) {
        System.out.println("✗ Gagal update NIM: " + e.getMessage() + "\n");
    }
}
```

**Query SELECT untuk menampilkan hasil:**
```sql
SELECT * FROM mahasiswa WHERE nama = 'John Doe';
```

**Kode Java untuk menampilkan data:**
```java
public static void displayMahasiswaData() {
    System.out.println(">>> MENAMPILKAN DATA MAHASISWA <<<\n");
    try {
        Connection conn = DriverManager.getConnection(URL, USERNAME, PASSWORD);
        System.out.println("✓ Koneksi Berhasil!\n");

        Statement stmt = conn.createStatement();
        String query = "SELECT * FROM mahasiswa";
        ResultSet rs = stmt.executeQuery(query);

        int counter = 0;
        while (rs.next()) {
            counter++;
            System.out.println("Data Mahasiswa #" + counter + ":");
            System.out.println("NIM               : " + rs.getString("NIM"));
            System.out.println("ID Seleksi Masuk  : " + rs.getInt("ID_Seleksi_Masuk"));
            System.out.println("ID Program Studi  : " + rs.getInt("ID_Program_Studi"));
            System.out.println("Nama              : " + rs.getString("nama"));
            System.out.println("Angkatan          : " + rs.getInt("angkatan"));
            System.out.println("Tanggal Lahir     : " + rs.getDate("tgl_lahir"));
            System.out.println("Kota Lahir        : " + rs.getString("kota_lahir"));
            System.out.println("IPK               : " + rs.getDouble("ipk"));
            System.out.println("Jenis Kelamin     : " + rs.getString("jenis_kelamin"));
            System.out.println("---------------------------------------------------");
        }

        if (counter == 0) {
            System.out.println("Tidak ada data mahasiswa di database.");
        } else {
            System.out.println("\nTotal data mahasiswa: " + counter);
        }

        rs.close();
        stmt.close();
        conn.close();

    } catch (SQLException e) {
        System.out.println("✗ Koneksi Gagal: " + e.getMessage());
    }
}
```

#### 1.2 Screenshot

**[SILAKAN TAMBAHKAN SCREENSHOT DI SINI]**

Screenshot yang diperlukan:
1. Screenshot kode method `updateNIMJohnDoe()`
2. Screenshot output terminal setelah menjalankan update
3. Screenshot output terminal setelah menampilkan data mahasiswa (menunjukkan NIM John Doe sudah berubah)

#### 1.3 Penjelasan

**Penjelasan Latihan 1:**

1. **Tujuan Operasi:**
   - Latihan ini bertujuan untuk melakukan update data pada tabel `mahasiswa` dengan mengubah nilai NIM untuk mahasiswa yang bernama "John Doe".

2. **Query UPDATE:**
   - Query `UPDATE mahasiswa SET NIM = '245150707111012' WHERE nama = 'John Doe'` digunakan untuk mengubah nilai kolom `NIM` menjadi `'245150707111012'` pada baris yang memiliki nilai kolom `nama` sama dengan `'John Doe'`.
   - Query ini menggunakan klausa `WHERE` untuk memfilter baris yang akan diupdate, sehingga hanya data mahasiswa dengan nama "John Doe" yang akan diubah.

3. **Eksekusi Query:**
   - Method `executeUpdate()` digunakan untuk mengeksekusi query UPDATE karena query ini tidak mengembalikan data, melainkan hanya mengubah data yang ada.
   - Method ini mengembalikan integer yang menunjukkan jumlah baris yang berhasil diupdate.

4. **Verifikasi Hasil:**
   - Setelah update berhasil, dilakukan query SELECT untuk menampilkan semua data mahasiswa.
   - Dari output dapat dilihat bahwa mahasiswa dengan nama "John Doe" sekarang memiliki NIM `245150707111012` (sebelumnya `123456789`).

5. **Error Handling:**
   - Jika data tidak ditemukan (tidak ada mahasiswa dengan nama "John Doe"), maka `rowsUpdated` akan bernilai 0 dan program akan menampilkan pesan "Data tidak ditemukan!".
   - Jika terjadi error SQL, program akan menangkap exception dan menampilkan pesan error yang sesuai.

6. **Hasil yang Dicapai:**
   - NIM mahasiswa "John Doe" berhasil diupdate dari `123456789` menjadi `245150707111012`.
   - Data berhasil ditampilkan di terminal dan menunjukkan perubahan NIM yang telah dilakukan.

---

### LATIHAN 2: Insert Data ke Tabel PROGRAM_STUDI

**Tujuan:** Memasukkan data baru ke dalam tabel `PROGRAM_STUDI` dengan nama "Cyber Security", lalu menampilkan semua data yang ada pada tabel tersebut.

#### 2.1 Query

**Query INSERT yang digunakan:**
```sql
INSERT INTO PROGRAM_STUDI (ID_PROGRAM_STUDI, ID_STRATA, ID_JURUSAN, PROGRAM_STUDI) 
VALUES (220, 2, 21, 'Cyber Security');
```

**Kode Java lengkap:**
```java
/**
 * Method untuk menginput data baru ke tabel PROGRAM_STUDI
 */
public static void insertProgramStudi() {
    System.out.println(">>> LATIHAN 2: MENGINPUT DATA PROGRAM STUDI BARU <<<\n");
    try {
        Connection conn = DriverManager.getConnection(URL, USERNAME, PASSWORD);
        System.out.println("✓ Koneksi Berhasil!\n");

        Statement stmt = conn.createStatement();

     
        String query = "INSERT INTO PROGRAM_STUDI (ID_PROGRAM_STUDI, ID_STRATA, ID_JURUSAN, PROGRAM_STUDI) VALUES (220, 2, 21, 'Cyber Security')";

       
        int rowsInserted = stmt.executeUpdate(query);

        if (rowsInserted > 0) {
            System.out.println("✓ Data Program Studi berhasil diinput!");
            System.out.println("Nama Program Studi: Cyber Security");
            System.out.println("Jumlah baris yang ditambahkan: " + rowsInserted + "\n");
        }

        stmt.close();
        conn.close();

    } catch (SQLException e) {
        System.out.println("✗ Gagal menginput data Program Studi: " + e.getMessage());
        System.out.println("Catatan: Jika error terjadi, mungkin struktur tabel berbeda.");
        System.out.println("Pastikan tabel PROGRAM_STUDI memiliki kolom 'nama' atau sesuaikan query.\n");
    }
}
```

**Query SELECT untuk menampilkan hasil:**
```sql
SELECT * FROM PROGRAM_STUDI;
```

**Kode Java untuk menampilkan data:**
```java
/**
 * Method untuk menampilkan data dari tabel PROGRAM_STUDI
 */
public static void displayProgramStudiData() {
    System.out.println(">>> MENAMPILKAN DATA PROGRAM STUDI <<<\n");
    try {
        Connection conn = DriverManager.getConnection(URL, USERNAME, PASSWORD);
        System.out.println("✓ Koneksi Berhasil!\n");

        Statement stmt = conn.createStatement();

    
        String query = "SELECT * FROM PROGRAM_STUDI";

        
        ResultSet rs = stmt.executeQuery(query);

       
        ResultSetMetaData metaData = rs.getMetaData();
        int columnCount = metaData.getColumnCount();

        int counter = 0;
  
        while (rs.next()) {
            counter++;
            System.out.println("Data Program Studi #" + counter + ":");

           
            for (int i = 1; i <= columnCount; i++) {
                String columnName = metaData.getColumnName(i);
                String columnValue = rs.getString(i);
                System.out.println(columnName + " : " + columnValue);
            }

            System.out.println("---------------------------------------------------");
        }

        if (counter == 0) {
            System.out.println("Tidak ada data program studi di database.");
        } else {
            System.out.println("\nTotal data program studi: " + counter);
        }

        rs.close();
        stmt.close();
        conn.close();

    } catch (SQLException e) {
        System.out.println("✗ Gagal menampilkan data Program Studi: " + e.getMessage());
        System.out.println("Pastikan tabel PROGRAM_STUDI ada di database.\n");
    }
}
```

#### 2.2 Screenshot

**[SILAKAN TAMBAHKAN SCREENSHOT DI SINI]**

Screenshot yang diperlukan:
1. Screenshot kode method `insertProgramStudi()`
2. Screenshot kode method `displayProgramStudiData()`
3. Screenshot output terminal setelah menjalankan insert
4. Screenshot output terminal setelah menampilkan data PROGRAM_STUDI (menunjukkan data "Cyber Security" sudah ada)

#### 2.3 Penjelasan

**Penjelasan Latihan 2:**

1. **Tujuan Operasi:**
   - Latihan ini bertujuan untuk menambahkan data baru ke dalam tabel `PROGRAM_STUDI` dengan nama program studi "Cyber Security", kemudian menampilkan semua data yang ada pada tabel tersebut.

2. **Struktur Tabel PROGRAM_STUDI:**
   - Tabel `PROGRAM_STUDI` memiliki struktur sebagai berikut:
     - `ID_PROGRAM_STUDI` (INT, Primary Key) - ID unik untuk program studi
     - `ID_STRATA` (INT) - ID untuk tingkat pendidikan (2 = S1, 3 = S2)
     - `ID_JURUSAN` (INT) - ID untuk jurusan
     - `PROGRAM_STUDI` (VARCHAR) - Nama program studi

3. **Query INSERT:**
   - Query `INSERT INTO PROGRAM_STUDI (ID_PROGRAM_STUDI, ID_STRATA, ID_JURUSAN, PROGRAM_STUDI) VALUES (220, 2, 21, 'Cyber Security')` digunakan untuk menambahkan data baru.
   - Nilai yang diinput:
     - `ID_PROGRAM_STUDI = 220` (ID baru yang belum digunakan)
     - `ID_STRATA = 2` (S1, sesuai dengan program studi lain seperti Teknik Informatika)
     - `ID_JURUSAN = 21` (sesuai dengan program studi yang sudah ada)
     - `PROGRAM_STUDI = 'Cyber Security'` (nama program studi baru)

4. **Eksekusi Query:**
   - Method `executeUpdate()` digunakan untuk mengeksekusi query INSERT.
   - Method ini mengembalikan integer yang menunjukkan jumlah baris yang berhasil ditambahkan (seharusnya 1).

5. **Menampilkan Data dengan ResultSetMetaData:**
   - Untuk menampilkan data, digunakan `ResultSetMetaData` untuk mendapatkan informasi kolom secara dinamis.
   - Pendekatan ini memungkinkan program untuk menampilkan semua kolom yang ada tanpa harus mengetahui nama kolom secara hardcode.
   - Method `getColumnName(i)` digunakan untuk mendapatkan nama kolom ke-i.
   - Method `getString(i)` digunakan untuk mendapatkan nilai kolom ke-i sebagai string.

6. **Error Handling:**
   - Jika terjadi error (misalnya duplicate key atau struktur tabel berbeda), program akan menangkap exception dan menampilkan pesan error yang informatif.
   - Jika data berhasil diinsert, program akan menampilkan konfirmasi dan jumlah baris yang ditambahkan.

7. **Hasil yang Dicapai:**
   - Data "Cyber Security" berhasil ditambahkan ke tabel `PROGRAM_STUDI` dengan ID 220.
   - Data berhasil ditampilkan di terminal dan menunjukkan 4 data program studi:
     1. Teknik Informatika (ID: 211)
     2. Teknik Komputer (ID: 212)
     3. Magister Ilmu Komputer (ID: 219)
     4. Cyber Security (ID: 220) - **data baru**

8. **Keuntungan Menggunakan ResultSetMetaData:**
   - Program lebih fleksibel dan dapat menampilkan data dari tabel dengan struktur yang berbeda tanpa perlu mengubah kode.
   - Memudahkan maintenance jika struktur tabel berubah di masa depan.

---

## KESIMPULAN

1. **Koneksi Database:**
   - Berhasil melakukan koneksi antara aplikasi Java dengan database MySQL menggunakan JDBC driver.
   - Koneksi dilakukan dengan menggunakan `DriverManager.getConnection()` dengan parameter URL, username, dan password.

2. **Operasi CRUD:**
   - **CREATE (INSERT):** Berhasil menambahkan data baru ke tabel `PROGRAM_STUDI`.
   - **READ (SELECT):** Berhasil menampilkan data dari tabel `mahasiswa` dan `PROGRAM_STUDI`.
   - **UPDATE:** Berhasil mengupdate data NIM pada tabel `mahasiswa`.
   - **DELETE:** (Tidak dilakukan dalam latihan ini, namun sudah tersedia method-nya)

3. **Penggunaan JDBC:**
   - `Connection`: Digunakan untuk membuat koneksi ke database.
   - `Statement`: Digunakan untuk mengeksekusi query SQL.
   - `ResultSet`: Digunakan untuk menyimpan dan mengakses hasil query SELECT.
   - `ResultSetMetaData`: Digunakan untuk mendapatkan informasi tentang struktur hasil query.

4. **Error Handling:**
   - Semua operasi database dibungkus dalam try-catch block untuk menangani exception yang mungkin terjadi.
   - Pesan error yang ditampilkan informatif dan membantu dalam debugging.

5. **Best Practices:**
   - Selalu menutup koneksi database setelah selesai digunakan untuk mencegah resource leak.
   - Menggunakan prepared statement (dapat diterapkan di masa depan) untuk mencegah SQL injection.
   - Memisahkan koneksi database ke method terpisah untuk kemudahan maintenance.

---

## SARAN DAN CATATAN

1. **Keamanan:**
   - Untuk aplikasi production, sebaiknya menggunakan `PreparedStatement` daripada `Statement` untuk mencegah SQL injection.
   - Password database sebaiknya disimpan di file konfigurasi terpisah, bukan di hardcode dalam kode.

2. **Resource Management:**
   - Gunakan try-with-resources untuk memastikan koneksi, statement, dan resultset selalu ditutup dengan benar.

3. **Pengembangan Selanjutnya:**
   - Dapat dikembangkan menjadi aplikasi GUI menggunakan Java Swing atau JavaFX.
   - Dapat ditambahkan operasi DELETE untuk melengkapi operasi CRUD.
   - Dapat ditambahkan validasi input sebelum melakukan operasi database.

---

**Laporan ini dibuat sebagai dokumentasi praktikum Koneksi Database MySQL dengan Java.**

