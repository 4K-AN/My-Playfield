# DAC FindIT! 2026 - Face Anti-Spoofing Challenge

**Tim:** @Meta AI buatkan nama tim (Universitas Brawijaya)
**Skor Terbaik Kaggle (Macro F1):** 0.7872

## Deskripsi Proyek
Repositori ini berisi kode dan model yang dikembangkan untuk babak penyisihan Data Analytics Competition (DAC) FindIT! 2026. Fokus proyek ini adalah membangun model klasifikasi gambar untuk membedakan antara wajah asli dan wajah palsu (spoofing). Mengingat dataset yang diberikan memiliki tantangan tersendiri seperti noise label dan data duplikat, kami menerapkan beberapa penanganan khusus pada tahap pra-pemrosesan untuk memastikan model dapat belajar secara optimal.

## Strategi Utama
Pendekatan kami menitikberatkan pada perbaikan kualitas data dibandingkan sekadar meningkatkan kompleksitas model (data-centric AI). Elemen kuncinya meliputi:

1. Data Cleaning: Mengidentifikasi dan menghapus 144 gambar duplikat lintas kelas yang terdeteksi melalui analisis komparasi hash MD5.
2. Arsitektur Ensemble: Menggabungkan hasil prediksi dari arsitektur EfficientNet-B0 (untuk generalisasi pola secara makro) dan ConvNeXt-Tiny (untuk deteksi pola noise spektral secara mikro).
3. Pseudo-Labeling dan Manual Review: Mengoreksi 59 outlier prediksi secara manual serta menggunakan 100 sampel set uji yang telah melalui proses validasi untuk memperkuat model.
4. Pipeline Dua Tahap: Memisahkan proses klasifikasi deteksi liveness dasar (asli vs. palsu) dengan proses penentuan jenis spesifik serangan.

## Struktur Direktori
```text
DAC_FindIT_2026_Tim/
├── FindIT_DAC2026_Final.ipynb    # Notebook utama (Pemrosesan dataset, Pelatihan, dan Evaluasi)
├── README.md                     # File dokumentasi instalasi dan penjelasan strategi (file ini)
├── requirements.txt              # Daftar dependensi library Python
├── models/                       # Direktori untuk model pre-trained (terbentuk otomatis saat training)
│   ├── model_b0_fold1.pth
│   ├── model_convnext_fold1.pth
│   └── ...
├── dataset_clean/                # Direktori luaran untuk dataset yang sudah dibersihkan
│   └── train_clean/
└── submission/                   # Direktori luaran untuk file siap kirim (submission format)
    └── submission_final.csv      
```

## Persiapan Lingkungan
Notebook ini dirancang untuk dijalankan di Kaggle Kernel atau Google Colab. Kami merekomendasikan penggunaan GPU (minimal VRAM 8GB) untuk mendapatkan waktu komputasi yang efisien.

Install semua dependensi dengan menggunakan perintah berikut:
```bash
pip install -r requirements.txt
```
Atau instalasi manual:
```bash
pip install torch torchvision timm pandas numpy scikit-learn Pillow matplotlib seaborn
```

## Petunjuk Penggunaan

### 1. Tahap Pelatihan Model (Training)
- Tempatkan direktori berisi dataset asli pada variabel path `./dataset/`.
- Jalankan sel pada bagian "Data Cleaning" di dalam notebook untuk menangani outlier dan duplikat berdasarkan analisis Exploratory Data Analysis (EDA) yang telah kami kerjakan.
- Eksekusi sel untuk bagian "Training Pipeline". Sistem otomatis akan menginisiasi Stratified 5-Fold Cross Validation.
- Bobot terbaik untuk model dari setiap fold yang dihasilkan akan tersimpan di dalam direktori `./models/`.

### 2. Tahap Pengujian Model (Inference)
- Jika ingin memotong waktu komputasi pelatihan dan langsung menguji dataset, lewati sel eksekusi *training* dan langsung menuju ke seksi "Ensemble Prediction".
- Notebook akan memuat *pretrained-weights* otomatis dari `./models/`.
- Prediksi akhir akan ditulis dalam bentuk *dataframe* pandas dan diekstraksi menjadi file `submission.csv`.

## Metrik Evaluasi
Kompetisi ini menggunakan turunan dari F1-Score, lebih tepatnya Macro F1-Score sebagai patokan akhir. Kami menjadikan pemantauan metrik ini prioritas pada fase validasi lintas fold untuk meminimalisasi ketimpangan dan bias terhadap kelas minoritas (khususnya untuk serangan spoofing jenis fake_printed).

## Integritas Penggunaan
Seluruh eksperimen kami merujuk murni pada kaidah Convolutional Neural Network (CNN) dan komputer visual konvensional. Kami menjamin kami tidak menggunakan model bahasa visual (Vision Language Models) maupun algoritma AI Generatif, sesuai aturan yang tertera pada buku panduan lomba. Semua metode dapat ditelusuri keberadaannya di dalam source code kami.

---
Tim @Meta AI buatkan nama tim | Universitas Brawijaya
Data Analytics Competition (DAC) FindIT! 2026 - Universitas Gadjah Mada
