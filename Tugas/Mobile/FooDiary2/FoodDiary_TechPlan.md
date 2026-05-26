# Tech Plan & Spesifikasi Proyek: Aplikasi "Food Diary"

**Dokumen ini ditujukan untuk AI Agent sebagai panduan utama (System Prompt / Context) dalam mengembangkan aplikasi Android "Food Diary".**

---

## 1. Deskripsi Umum Proyek
- **Nama Aplikasi**: Food Diary
- **Platform**: Android
- **Fungsi Utama**: Jurnal digital dan manajemen inventarisasi kuliner pribadi untuk mendokumentasikan jejak kuliner (nama makanan, deskripsi, harga/ulasan, dan foto).
- **Pendekatan UI**: Deklaratif menggunakan **Jetpack Compose**.
- **Arsitektur**: **MVVM (Model-View-ViewModel)** dengan pemisahan logika bisnis dari antarmuka pengguna (State Hoisting).
- **Backend / Infrastruktur**: **Supabase** (BaaS)
  - **Supabase Auth**: Autentikasi Pengguna.
  - **Supabase Database (PostgreSQL)**: Manajemen data relasional (teks, relasi).
  - **Supabase Storage**: Penyimpanan berkas media (foto makanan). Pendekatan *URL-Based Storage*.

---

## 2. Struktur Basis Data (Supabase Schema)
Terdapat tiga entitas utama yang saling berelasi:

### A. Tabel `profiles` (Relasi 1:1 dengan `auth.users` Supabase)
- `id` (UUID): Primary Key & Foreign Key merujuk ke `auth.users(id)`. Cascade delete.
- `full_name` (VARCHAR 255): Nama lengkap pengguna.
- `avatar_url` (TEXT): URL foto profil dari Storage.
- `updated_at` (TIMESTAMPTZ): Waktu pembaruan.

### B. Tabel `categories` (Global)
- `id` (UUID): Primary Key, default `gen_random_uuid()`.
- `name` (VARCHAR 100): Nama kategori (UNIQUE).
- `created_at` (TIMESTAMPTZ): Waktu pembuatan.

### C. Tabel `food_items` (Entitas Utama CRUD)
- `id` (UUID): Primary Key, default `gen_random_uuid()`.
- `user_id` (UUID): Foreign Key merujuk ke `profiles(id)`. Cascade delete. (Filter data agar user hanya melihat datanya sendiri).
- `category_id` (UUID, Nullable): Foreign Key merujuk ke `categories(id)`. Set null on delete.
- `title` (VARCHAR 255): Judul/nama makanan.
- `description` (TEXT): Deskripsi detail/ulasan makanan.
- `image_url` (TEXT): URL akses publik gambar dari Supabase Storage (Bucket: `food-images`).
- `created_at` (TIMESTAMPTZ): Waktu pembuatan data.

---

## 3. Alur Navigasi (Navigation Compose Workflow)
Routing aplikasi harus mencakup rute berikut:

1. **Splash/Init Route**: Cek sesi *Current User* di Supabase.
   - Jika `null` -> Navigasi ke **AuthScreen**.
   - Jika ada sesi aktif -> Navigasi ke **HomeScreen**.
2. **Auth Route**: Layar Login/Register. Jika sukses -> Pop dan arahkan ke **HomeScreen**.
3. **Home Route**: Menampilkan daftar makanan. 
   - Aksi: FAB diklik -> ke **FormScreen** (Mode Insert).
   - Aksi: Item diklik -> ke **DetailScreen** (membawa argumen ID).
   - Aksi: Ikon Profil diklik -> ke **ProfileScreen**.
4. **Detail Route**: Menampilkan detail lengkap.
   - Aksi: Tombol Edit -> Navigasi ke **FormScreen** (Mode Update, membawa argumen ID).
   - Aksi: Tombol Hapus -> Eksekusi delete, pop kembali ke **HomeScreen**.
5. **Form Route**: Form input (teks dan gambar). Sukses simpan -> Pop kembali ke **HomeScreen**.
6. **Profile Route**: Menampilkan info user. Aksi: Logout -> Hapus sesi, arahkan ke **AuthScreen**.

---

## 4. Spesifikasi Antarmuka Pengguna (UI) per Layar

### A. AuthScreen (Login & Register)
- **Layout**: `Column` terpusat (`CenterVertically` & `CenterHorizontally`).
- **Komponen Inti**: 
  - `OutlinedTextField` untuk Email & Password.
  - `Button` untuk Submit.
  - `TextButton` untuk Toggle mode Login / Register.
- **State Handling**: Gunakan `LaunchedEffect` untuk memantau transisi state (Loading, Success, Error) dari ViewModel.

### B. HomeScreen (List Dashboard)
- **Layout**: `Scaffold`.
- **Komponen Inti**:
  - `TopAppBar`: Berisi judul aplikasi dan ikon profil (navigasi ke ProfileScreen).
  - `LazyVerticalGrid` dengan `GridCells.Adaptive(150.dp)`: Menampilkan daftar item dalam bentuk `Card` secara responsif.
  - `FloatingActionButton`: Ikon (+) untuk navigasi ke FormScreen.

### C. DetailScreen
- **Layout**: `Scaffold` dengan `Column` (tambahkan modifier `.verticalScroll(rememberScrollState())` agar deskripsi panjang bisa digulir).
- **Komponen Inti**:
  - `AsyncImage` (Library Coil): Merender gambar dari `image_url`.
  - `Text`: Format Headline (Judul) dan Body (Deskripsi).
  - `Row`: Berisi dua `Button` (Edit dan Delete).

### D. FormScreen (Create / Update)
- **Layout**: `Column` vertikal.
- **Komponen Inti**:
  - **Area Image Picker**: Komponen `Box` yang dapat diklik. Memanggil `ActivityResultLauncher` untuk memilih gambar dari galeri perangkat.
  - `AsyncImage`: Menampilkan preview lokal dari URI gambar yang dipilih.
  - `OutlinedTextField`: Untuk input `title` dan `description`.
  - `Button` (Simpan): Terhubung ke fungsi `viewModel.upsertData()`.
- **Alur Simpan Data**: Gambar diunggah ke Supabase Storage terlebih dahulu -> Dapatkan Public URL -> Simpan URL dan data teks ke Supabase Database.

### E. ProfileScreen
- **Layout**: `Column` terpusat.
- **Komponen Inti**:
  - Ikon Avatar besar.
  - `Text`: Menampilkan email dari Supabase *Current User*.
  - `Button` (Warna Merah): Fungsi Logout.

---

## 5. Kebutuhan Library / Dependencies Tambahan
1. **Supabase-kt**: Untuk integrasi Auth, PostgREST (Database), dan Storage.
2. **Ktor Client**: Engine HTTP yang dibutuhkan oleh Supabase-kt.
3. **Coil-Compose**: Untuk memuat gambar asinkron (`AsyncImage`) dari URL Supabase Storage.
4. **Navigation Compose**: Untuk sistem routing halaman.
5. **ViewModel & Lifecycle**: Untuk arsitektur MVVM dan `collectAsStateWithLifecycle`.
