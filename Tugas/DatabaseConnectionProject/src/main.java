import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.SQLException;
import java.sql.Statement;

public class main {
    // MySQL Connection Parameters
    // Sesuaikan port dan password dengan konfigurasi MySQL Anda
    private static final String URL = "jdbc:mysql://127.0.0.1:3306/akademik";
    private static final String USERNAME = "root";
    private static final String PASSWORD = "221205";

    public static void main(String[] args) {
        System.out.println("===============================================");
        System.out.println("APLIKASI KONEKSI DATABASE AKADEMIK DENGAN JAVA");
        System.out.println("===============================================\n");

        // Uncomment salah satu method di bawah untuk menjalankan operasi yang
        // diinginkan

        // 1. Test Koneksi Database
        testConnection();

        // LATIHAN 1: Update NIM John Doe menjadi 245150707111012
        updateNIMJohnDoe();
        
        // Tampilkan data mahasiswa setelah update
        displayMahasiswaData();

        // LATIHAN 2: Insert data baru ke PROGRAM_STUDI
        insertProgramStudi();
        
        // Tampilkan data PROGRAM_STUDI
        displayProgramStudiData();

        // 2. Menampilkan Data Mahasiswa (Uncomment untuk menjalankan)
        // displayMahasiswaData();

        // 3. Menginput Data Mahasiswa Baru (Uncomment untuk menjalankan)
        // insertMahasiswaData();

        // 4. Update Data Mahasiswa (Uncomment untuk menjalankan)
        // updateMahasiswaData();

        // 5. Delete Data Mahasiswa (Uncomment untuk menjalankan)
        // deleteMahasiswaData();
    }

    /**
     * Method untuk menguji koneksi ke database MySQL
     */
    public static void testConnection() {
        System.out.println(">>> TEST KONEKSI DATABASE <<<\n");
        try {
            Connection conn = DriverManager.getConnection(URL, USERNAME, PASSWORD);
            System.out.println("✓ Koneksi Berhasil!");
            System.out.println("Database: akademik");
            System.out.println("Host: localhost");
            System.out.println("Port: 3306");
            conn.close();
        } catch (SQLException e) {
            System.out.println("✗ Koneksi Gagal: " + e.getMessage());
            System.out.println("Pastikan:");
            System.out.println("1. MySQL Server berjalan");
            System.out.println("2. Database 'akademik' sudah dibuat");
            System.out.println("3. Username dan password sesuai");
            System.out.println("4. Port MySQL sesuai (default: 3306)");
        }
    }

    /**
     * Method untuk menampilkan data mahasiswa dari database
     */
    public static void displayMahasiswaData() {
        System.out.println(">>> MENAMPILKAN DATA MAHASISWA <<<\n");
        try {
            Connection conn = DriverManager.getConnection(URL, USERNAME, PASSWORD);
            System.out.println("✓ Koneksi Berhasil!\n");

            // Membuat Statement untuk eksekusi query
            Statement stmt = conn.createStatement();

            // Query SELECT untuk menampilkan semua data mahasiswa
            String query = "SELECT * FROM mahasiswa";

            // Eksekusi query dan menyimpan hasil dalam ResultSet
            ResultSet rs = stmt.executeQuery(query);

            int counter = 0;
            // Iterasi setiap baris dalam ResultSet
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

    /**
     * Method untuk menginput data mahasiswa baru ke database
     */
    public static void insertMahasiswaData() {
        System.out.println(">>> MENGINPUT DATA MAHASISWA BARU <<<\n");
        try {
            Connection conn = DriverManager.getConnection(URL, USERNAME, PASSWORD);
            System.out.println("✓ Koneksi Berhasil!\n");

            Statement stmt = conn.createStatement();

            // Query INSERT untuk menambah data mahasiswa baru
            String query = "INSERT INTO mahasiswa (NIM, ID_Seleksi_Masuk, ID_Program_Studi, "
                    + "nama, angkatan, tgl_lahir, kota_lahir, jenis_kelamin) "
                    + "VALUES ('123456789', 1, 211, 'John Doe', 2023, '2000-01-01', 'Jakarta', 'P')";

            // Eksekusi query INSERT
            int rowsInserted = stmt.executeUpdate(query);

            if (rowsInserted > 0) {
                System.out.println("✓ Data berhasil diinput!");
                System.out.println("Jumlah baris yang ditambahkan: " + rowsInserted);
                System.out.println("\nData yang diinput:");
                System.out.println("NIM: 123456789");
                System.out.println("Nama: John Doe");
                System.out.println("Angkatan: 2023");
            }

            stmt.close();
            conn.close();

        } catch (SQLException e) {
            System.out.println("✗ Gagal menginput data: " + e.getMessage());
        }
    }

    /**
     * Method untuk mengupdate data mahasiswa
     */
    public static void updateMahasiswaData() {
        System.out.println(">>> UPDATE DATA MAHASISWA <<<\n");
        try {
            Connection conn = DriverManager.getConnection(URL, USERNAME, PASSWORD);
            System.out.println("✓ Koneksi Berhasil!\n");

            Statement stmt = conn.createStatement();

            // Query UPDATE untuk mengubah data mahasiswa
            String query = "UPDATE mahasiswa SET ipk = 3.75, kota_lahir = 'Surabaya' " + "WHERE NIM = '123456789'";

            // Eksekusi query UPDATE
            int rowsUpdated = stmt.executeUpdate(query);

            if (rowsUpdated > 0) {
                System.out.println("✓ Data berhasil diupdate!");
                System.out.println("Jumlah baris yang diubah: " + rowsUpdated);
            } else {
                System.out.println("✗ Data tidak ditemukan!");
            }

            stmt.close();
            conn.close();

        } catch (SQLException e) {
            System.out.println("✗ Gagal update data: " + e.getMessage());
        }
    }

    /**
     * Method untuk menghapus data mahasiswa
     */
    public static void deleteMahasiswaData() {
        System.out.println(">>> DELETE DATA MAHASISWA <<<\n");
        try {
            Connection conn = DriverManager.getConnection(URL, USERNAME, PASSWORD);
            System.out.println("✓ Koneksi Berhasil!\n");

            Statement stmt = conn.createStatement();

            // Query DELETE untuk menghapus data mahasiswa
            String query = "DELETE FROM mahasiswa WHERE NIM = '123456789'";

            // Eksekusi query DELETE
            int rowsDeleted = stmt.executeUpdate(query);

            if (rowsDeleted > 0) {
                System.out.println("✓ Data berhasil dihapus!");
                System.out.println("Jumlah baris yang dihapus: " + rowsDeleted);
            } else {
                System.out.println("✗ Data tidak ditemukan!");
            }

            stmt.close();
            conn.close();

        } catch (SQLException e) {
            System.out.println("✗ Gagal menghapus data: " + e.getMessage());
        }
    }

    /**
     * LATIHAN 1: Method untuk mengupdate NIM mahasiswa John Doe menjadi 245150707111012
     */
    public static void updateNIMJohnDoe() {
        System.out.println(">>> LATIHAN 1: UPDATE NIM JOHN DOE <<<\n");
        try {
            Connection conn = DriverManager.getConnection(URL, USERNAME, PASSWORD);
            System.out.println("✓ Koneksi Berhasil!\n");

            Statement stmt = conn.createStatement();

            String query = "UPDATE mahasiswa SET NIM = '245150707111012' WHERE nama = 'John Doe'";

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

    /**
     * LATIHAN 2: Method untuk menginput data baru ke tabel PROGRAM_STUDI
     */
    public static void insertProgramStudi() {
        System.out.println(">>> LATIHAN 2: MENGINPUT DATA PROGRAM STUDI BARU <<<\n");
        try {
            Connection conn = DriverManager.getConnection(URL, USERNAME, PASSWORD);
            System.out.println("✓ Koneksi Berhasil!\n");

            Statement stmt = conn.createStatement();

            // Query INSERT untuk menambah data program studi baru
            // Struktur tabel: ID_PROGRAM_STUDI, ID_STRATA, ID_JURUSAN, PROGRAM_STUDI
            // Menggunakan ID_PROGRAM_STUDI=220 (belum digunakan), ID_STRATA=2 (S1), ID_JURUSAN=21
            String query = "INSERT INTO PROGRAM_STUDI (ID_PROGRAM_STUDI, ID_STRATA, ID_JURUSAN, PROGRAM_STUDI) VALUES (220, 2, 21, 'Cyber Security')";

            // Eksekusi query INSERT
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

    /**
     * LATIHAN 2: Method untuk menampilkan data dari tabel PROGRAM_STUDI
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
}
