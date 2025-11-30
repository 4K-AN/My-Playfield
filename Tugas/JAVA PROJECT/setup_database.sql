-- ================================================
-- Script Setup Database Akademik
-- Modul 12: Pemrograman SQL
-- ================================================

-- 1. Membuat Database
CREATE DATABASE IF NOT EXISTS akademik;
USE akademik;

-- 2. Membuat Tabel Mahasiswa
CREATE TABLE IF NOT EXISTS mahasiswa (
    NIM VARCHAR(20) PRIMARY KEY,
    ID_Seleksi_Masuk INT NOT NULL,
    ID_Program_Studi INT NOT NULL,
    nama VARCHAR(100) NOT NULL,
    angkatan INT NOT NULL,
    tgl_lahir DATE,
    kota_lahir VARCHAR(50),
    jenis_kelamin CHAR(1),
    ipk DECIMAL(3,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. Insert Data Sample (Optional)
INSERT INTO mahasiswa (NIM, ID_Seleksi_Masuk, ID_Program_Studi, nama, angkatan, tgl_lahir, kota_lahir, jenis_kelamin, ipk) 
VALUES 
('001', 1, 211, 'Ahmad Hidayat', 2023, '2003-05-15', 'Jakarta', 'L', 3.45),
('002', 1, 211, 'Siti Nurhaliza', 2023, '2003-08-20', 'Bandung', 'P', 3.65),
('003', 2, 212, 'Budi Santoso', 2023, '2003-03-10', 'Surabaya', 'L', 3.25),
('004', 1, 211, 'Rina Kusuma', 2023, '2003-11-25', 'Medan', 'P', 3.80),
('005', 2, 212, 'Hendra Wijaya', 2023, '2003-01-30', 'Yogyakarta', 'L', 3.15);

-- 4. Verifikasi Data
SELECT * FROM mahasiswa;

-- ================================================
-- Query Berguna untuk Testing
-- ================================================

-- Tampilkan semua mahasiswa
-- SELECT * FROM mahasiswa;

-- Tampilkan mahasiswa dengan IPK > 3.5
-- SELECT * FROM mahasiswa WHERE ipk > 3.5;

-- Tampilkan mahasiswa berdasarkan jenis kelamin
-- SELECT * FROM mahasiswa WHERE jenis_kelamin = 'P';

-- Tampilkan mahasiswa dengan urutan IPK descending
-- SELECT * FROM mahasiswa ORDER BY ipk DESC;

-- Tampilkan jumlah mahasiswa per angkatan
-- SELECT angkatan, COUNT(*) as jumlah FROM mahasiswa GROUP BY angkatan;

-- Tampilkan mahasiswa dari kota tertentu
-- SELECT * FROM mahasiswa WHERE kota_lahir = 'Jakarta';

-- ================================================
-- Cleanup (jika diperlukan)
-- ================================================

-- DROP TABLE IF EXISTS mahasiswa;
-- DROP DATABASE IF EXISTS akademik;
