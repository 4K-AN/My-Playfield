# Cara Menjalankan Aplikasi

## Masalah yang Sering Terjadi

IDE (Cursor/VS Code) secara otomatis mengcompile file Java dengan Java 25 (yang terinstall di sistem), padahal runtime IDE adalah Java 21. Ini menyebabkan error:
```
UnsupportedClassVersionError: main has been compiled by a more recent version of the Java Runtime (class file version 69.0)
```

## Solusi

### Opsi 1: Gunakan Run Configuration dengan Pre-Launch Task (RECOMMENDED)

1. Tekan `F5` atau klik "Run and Debug"
2. Pilih **"main (with pre-compile)"** dari dropdown
3. Aplikasi akan otomatis:
   - Clean file .class lama
   - Compile dengan target Java 17
   - Run aplikasi

### Opsi 2: Gunakan Task Manual

1. Tekan `Ctrl+Shift+P`
2. Ketik "Tasks: Run Task"
3. Pilih **"java: build"** (akan clean dan compile)
4. Lalu pilih **"java: run"** untuk menjalankan

### Opsi 3: Gunakan Script Batch

Jalankan langsung dari terminal:
```cmd
run.bat
```

Atau:
```cmd
compile.bat
java -cp target/classes;mysql-connector-j-9.5.0/mysql-connector-j-9.5.0.jar main
```

### Opsi 4: Manual Compile Sebelum Run

Jika masih error, jalankan ini di terminal sebelum run:
```powershell
Remove-Item target\classes\*.class -Force -ErrorAction SilentlyContinue
javac --release 17 -cp mysql-connector-j-9.5.0/mysql-connector-j-9.5.0.jar -d target/classes src/main.java
```

Lalu run dari IDE.

## Mengapa Ini Terjadi?

- Sistem memiliki Java 25 terinstall
- IDE extension Java menggunakan Java 21 sebagai runtime
- Ketika IDE auto-compile, menggunakan Java 25 (dari sistem)
- File .class yang dihasilkan tidak kompatibel dengan Java 21 runtime

## Solusi Permanen

Konfigurasi sudah diupdate untuk:
- ✅ Auto-clean file .class lama sebelum compile
- ✅ Selalu compile dengan target Java 17 (kompatibel dengan Java 21)
- ✅ Disable auto-build (harus manual atau via task)
- ✅ Pre-launch task untuk compile sebelum run

**Gunakan Run Configuration "main (with pre-compile)" untuk hasil terbaik!**

