# Dokumentasi API JWT - Lumen

## 📋 Daftar Isi
1. [Penambahan Data Dummy & Validasi Email Unik](#1-penambahan-data-dummy--validasi-email-unik)
2. [Pembuatan Endpoint Token Check](#2-pembuatan-endpoint-token-check)
3. [Skenario Pengujian 1: Register Berhasil](#3-skenario-pengujian-1-register-berhasil)
4. [Skenario Pengujian 2: Login Berhasil](#4-skenario-pengujian-2-login-berhasil)
5. [Skenario Pengujian 3: Login Gagal (Password Salah)](#5-skenario-pengujian-3-login-gagal-password-salah)
6. [Skenario Pengujian 4: Profile Berhasil](#6-skenario-pengujian-4-profile-berhasil)
7. [Skenario Pengujian 5: Logout Berhasil](#7-skenario-pengujian-5-logout-berhasil)
8. [Skenario Pengujian 6: Token Check Berhasil](#8-skenario-pengujian-6-token-check-berhasil)
9. [Dokumentasi Header JWT](#9-dokumentasi-header-jwt)

---

## 1. Penambahan Data Dummy & Validasi Email Unik

### 📝 Deskripsi
Menambahkan satu data user dummy baru pada array `$users` dan menambahkan validasi pada method `register()` agar email harus unik terhadap daftar user dummy yang sudah ada.

### 💻 Lokasi File
**File:** `app/Http/Controllers/Api/AuthController.php`

### 🔧 Implementasi Kode

#### A. Penambahan User Dummy Baru
```php
private $users = [
    [
        'id' => 1,
        'name' => 'User Cakep',
        'email' => 'user@example.com',
        'password' => 'password123'
    ],
    [
        'id' => 2,
        'name' => 'Admin Hebat',
        'email' => 'admin@example.com',
        'password' => 'secret321'
    ],
    [
        'id' => 3,
        'name' => 'Akhmad Syafiul Anam',
        'email' => 'syafi@student.ub.ac.id',
        'password' => 'rahasia123'
    ]
];
```

#### B. Penambahan Validasi Email Unik pada Method `register()`
```php
public function register(Request $request)
{
    $validated = $request->validate([
        'name' => 'required|string|max:100',
        'email' => 'required|email',
        'password' => 'required|string|min:6|confirmed'
    ]);

    // TUGAS 2: Validasi agar email harus unik terhadap daftar dummy
    $isEmailExists = collect($this->users)->firstWhere('email', $validated['email']);
    
    if ($isEmailExists) {
        return response()->json([
            'message' => 'Pendaftaran gagal, email sudah terdaftar di sistem.'
        ], 422);
    }

    $user = [
        'id' => rand(4, 1000),
        'name' => $validated['name'],
        'email' => $validated['email'],
        'password' => $validated['password'],
    ];

    return response()->json([
        'message' => 'User registered successfully (dummy)',
        'user' => $user
    ], 201);
}
```

### 📖 Penjelasan
- Karena sistem tidak menggunakan database, validasi `unique:users` bawaan Laravel tidak bisa digunakan
- Solusi: Memanfaatkan helper `collect()` untuk membungkus array dummy
- Fungsi `firstWhere()` mencari kecocokan email dalam array
- Jika email ditemukan, API memblokir pendaftaran dengan status **422 Unprocessable Entity**
- Jika email unik, user berhasil didaftarkan dengan status **201 Created**

---

## 2. Pembuatan Endpoint Token Check

### 📝 Deskripsi
Membuat endpoint `GET /api/token-check` yang hanya dapat diakses jika token valid, dan mengembalikan response JSON berisi pesan "Token valid" beserta data user.

### 💻 Lokasi File
- **Routes:** `routes/api.php`
- **Controller:** `app/Http/Controllers/Api/AuthController.php`

### 🔧 Implementasi Kode

#### A. Penambahan Route
```php
Route::middleware(['dummy.jwt'])->group(function () {
    Route::post('/logout', [AuthController::class, 'logout']);
    Route::get('/profile', [AuthController::class, 'profile']);
    
    // Tugas 3: Menambahkan endpoint token-check
    Route::get('/token-check', [AuthController::class, 'tokenCheck']);
});
```

#### B. Penambahan Method pada AuthController
```php
public function tokenCheck(Request $request)
{
    // Jika request berhasil masuk ke method ini, artinya token sudah pasti valid 
    // karena berhasil melewati DummyJwtMiddleware.
    $payload = $request->jwt_payload;
    
    return response()->json([
        'message' => 'Token valid',
        'user' => [
            'email' => $payload->get('email'),
            'name' => $payload->get('name')
        ]
    ], 200);
}
```

### 📖 Penjelasan
- Endpoint didaftarkan dalam grup middleware `dummy.jwt`
- Request tidak akan pernah mencapai method `tokenCheck()` jika token tidak valid
- Jika berhasil lolos middleware, method mengambil data dari `$request->jwt_payload`
- Response mengembalikan status **200 OK** dengan pesan "Token valid"

---

## 3. Skenario Pengujian 1: Register Berhasil

### 📝 Deskripsi
Melakukan pengujian skenario register berhasil dengan email yang belum terdaftar di sistem.

### 🔗 Endpoint
```
Method: POST
URL: http://localhost:8000/api/register
```

### 📤 Request Headers
```http
Content-Type: application/json
```

### 📥 Request Body
```json
{
    "name": "Mahasiswa Baru",
    "email": "maba@student.ub.ac.id",
    "password": "password123",
    "password_confirmation": "password123"
}
```

### ✅ Response (Status 201 Created)
```json
{
    "message": "User registered successfully (dummy)",
    "user": {
        "id": 523,
        "name": "Mahasiswa Baru",
        "email": "maba@student.ub.ac.id",
        "password": "password123"
    }
}
```

### 📖 Penjelasan
- Email `maba@student.ub.ac.id` belum ada di array dummy
- Validasi email unik lolos
- Sistem mengembalikan status **201 Created**
- Sistem menghasilkan ID random untuk user baru (4-1000)

---

## 4. Skenario Pengujian 2: Login Berhasil

### 📝 Deskripsi
Melakukan pengujian skenario login berhasil menggunakan kredensial dummy yang valid.

### 🔗 Endpoint
```
Method: POST
URL: http://localhost:8000/api/login
```

### 📤 Request Headers
```http
Content-Type: application/json
```

### 📥 Request Body
```json
{
    "email": "syafi@student.ub.ac.id",
    "password": "rahasia123"
}
```

### ✅ Response (Status 200 OK)
```json
{
    "message": "Login successful (dummy)",
    "token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJodHRwOi8vbG9jYWxob3N0OjgwMDAiLCJhdWQiOm51bGwsImlhdCI6MTcxNjAxNzA2MCwiZXhwIjoxNzE2MDIwNjYwLCJlbWFpbCI6InN5YWZpQHN0dWRlbnQudWIuYWMuaWQiLCJuYW1lIjoiQWtobWFkIFN5YWZpdWwgQW5hbSJ9.x_YZ9K2L4m_N5p_Q6r_S7t_U8v_W9x_Y0z_A1b_C2d"
}
```

### 📖 Penjelasan
- Login menggunakan kredensial dari user dummy ID 3
- Email dan password cocok dengan data di array `$users`
- Sistem melakukan generate JWT Token dengan klaim `email` dan `name`
- Status **200 OK** dikembalikan
- Token JWT ini akan digunakan untuk mengakses endpoint yang diproteksi

---

## 5. Skenario Pengujian 3: Login Gagal (Password Salah)

### 📝 Deskripsi
Melakukan pengujian skenario login gagal karena password tidak sesuai dengan data di database dummy.

### 🔗 Endpoint
```
Method: POST
URL: http://localhost:8000/api/login
```

### 📤 Request Headers
```http
Content-Type: application/json
```

### 📥 Request Body
```json
{
    "email": "syafi@student.ub.ac.id",
    "password": "passwordngawur"
}
```

### ❌ Response (Status 401 Unauthorized)
```json
{
    "message": "Invalid email or password"
}
```

### 📖 Penjelasan
- Email ditemukan di array dummy, tetapi password tidak cocok
- Password input `passwordngawur` ≠ password di data `rahasia123`
- Sistem mendeteksi ketidakcocokan dan mengembalikan status **401 Unauthorized**
- Pesan error tidak membedakan antara email tidak ditemukan atau password salah (best practice security)

---

## 6. Skenario Pengujian 4: Profile Berhasil

### 📝 Deskripsi
Melakukan pengujian skenario profile berhasil dengan token valid. Endpoint ini membuktikan bahwa token berhasil dibongkar oleh middleware.

### 🔗 Endpoint
```
Method: GET
URL: http://localhost:8000/api/profile
```

### 📤 Request Headers
```http
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJodHRwOi8vbG9jYWxob3N0OjgwMDAiLCJhdWQiOm51bGwsImlhdCI6MTcxNjAxNzA2MCwiZXhwIjoxNzE2MDIwNjYwLCJlbWFpbCI6InN5YWZpQHN0dWRlbnQudWIuYWMuaWQiLCJuYW1lIjoiQWtobWFkIFN5YWZpdWwgQW5hbSJ9.x_YZ9K2L4m_N5p_Q6r_S7t_U8v_W9x_Y0z_A1b_C2d
Accept: application/json
```

### 📥 Request Body
```
(Kosong - tidak ada body untuk GET request)
```

### ✅ Response (Status 200 OK)
```json
{
    "user": {
        "email": "syafi@student.ub.ac.id",
        "name": "Akhmad Syafiul Anam"
    }
}
```

### 📖 Penjelasan
- Request membawa token JWT valid di header Authorization
- Token berhasil melewati middleware `DummyJwtMiddleware`
- Middleware membongkar payload JWT dan menyimpannya di `$request->jwt_payload`
- Method `profile()` mengambil data email dan name dari payload
- Status **200 OK** dikembalikan dengan rincian user

---

## 7. Skenario Pengujian 5: Logout Berhasil

### 📝 Deskripsi
Melakukan pengujian skenario logout berhasil. Endpoint ini menginvalidasi/mem-blacklist token agar tidak bisa dipakai lagi.

### 🔗 Endpoint
```
Method: POST
URL: http://localhost:8000/api/logout
```

### 📤 Request Headers
```http
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJodHRwOi8vbG9jYWxob3N0OjgwMDAiLCJhdWQiOm51bGwsImlhdCI6MTcxNjAxNzA2MCwiZXhwIjoxNzE2MDIwNjYwLCJlbWFpbCI6InN5YWZpQHN0dWRlbnQudWIuYWMuaWQiLCJuYW1lIjoiQWtobWFkIFN5YWZpdWwgQW5hbSJ9.x_YZ9K2L4m_N5p_Q6r_S7t_U8v_W9x_Y0z_A1b_C2d
Accept: application/json
```

### 📥 Request Body
```
(Kosong - tidak ada body untuk POST logout)
```

### ✅ Response (Status 200 OK)
```json
{
    "message": "User logged out successfully"
}
```

### 📖 Penjelasan
- Endpoint `/api/logout` dijalankan sambil membawa token aktif
- Method `logout()` mengeksekusi `JWTAuth::invalidate()` untuk mem-blacklist token
- Token yang sudah di-logout tidak akan bisa digunakan lagi
- Status **200 OK** dikembalikan dengan pesan sukses

---

## 8. Skenario Pengujian 6: Token Check Berhasil

### 📝 Deskripsi
Melakukan pengujian endpoint token-check dengan token valid. Endpoint ini hanya untuk verifikasi bahwa token masih aktif dan valid.

### 🔗 Endpoint
```
Method: GET
URL: http://localhost:8000/api/token-check
```

### 📤 Request Headers
```http
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJodHRwOi8vbG9jYWxob3N0OjgwMDAiLCJhdWQiOm51bGwsImlhdCI6MTcxNjAxNzA2MCwiZXhwIjoxNzE2MDIwNjYwLCJlbWFpbCI6InN5YWZpQHN0dWRlbnQudWIuYWMuaWQiLCJuYW1lIjoiQWtobWFkIFN5YWZpdWwgQW5hbSJ9.x_YZ9K2L4m_N5p_Q6r_S7t_U8v_W9x_Y0z_A1b_C2d
Accept: application/json
```

### 📥 Request Body
```
(Kosong - tidak ada body untuk GET request)
```

### ✅ Response (Status 200 OK)
```json
{
    "message": "Token valid",
    "user": {
        "email": "syafi@student.ub.ac.id",
        "name": "Akhmad Syafiul Anam"
    }
}
```

### 📖 Penjelasan
- Endpoint ini khusus untuk memverifikasi validitas token
- Jika token tidak valid, middleware akan menolak request dengan status 401
- Jika token valid, endpoint mengembalikan status **200 OK** dengan pesan "Token valid"
- Berguna untuk aplikasi frontend untuk mengecek apakah token masih aktif

---

## 9. Dokumentasi Header JWT

### 📝 Deskripsi
Dokumentasi lengkap header yang wajib dikirim saat mengakses endpoint yang diproteksi JWT.

### 🔐 Header Requirement

#### Header Wajib untuk Endpoint Berproteksi JWT
```http
Authorization: Bearer <JWT_TOKEN_DI_SINI>
Accept: application/json
```

### 📋 Penjelasan Setiap Header

| Header | Nilai | Penjelasan |
|--------|-------|-----------|
| `Authorization` | `Bearer <token>` | Mengirim kredensial autentikasi. Kata "Bearer" adalah standar OAuth 2.0 yang memberitahu server untuk membaca token JWT. Token diperoleh dari endpoint `/api/login` |
| `Accept` | `application/json` | Memberitahu server bahwa client menerima response dalam format JSON |
| `Content-Type` | `application/json` | Diperlukan saat mengirim request body (POST, PUT, PATCH). Memberitahu server bahwa body dalam format JSON |

### 🔄 Alur Pengiriman Token

```
1. Client melakukan login ke /api/login
   ↓
2. Server mengembalikan JWT token
   ↓
3. Client menyimpan token (di localStorage, sessionStorage, atau cookie)
   ↓
4. Client mengirim token di Authorization header untuk setiap request ke endpoint berproteksi
   ↓
5. Middleware DummyJwtMiddleware memvalidasi token
   ↓
6. Jika valid → request dilanjutkan, jika tidak → return 401 Unauthorized
```

### 📤 Contoh Request Lengkap (Postman)

#### Method & URL
```
GET http://localhost:8000/api/profile
```

#### Tab: Headers
```
Key                 | Value                                        | Description
Authorization       | Bearer eyJ0eXAiOiJKV1QiLC...                 | Token JWT dari login
Accept              | application/json                             | Response format
```

#### Tab: Body
```
(Kosong untuk GET request)
```

### 💡 Tips untuk Developer

1. **Jangan pernah hardcode token** - Simpan token secara aman (localStorage/sessionStorage)
2. **Expiration handling** - Token memiliki masa berlaku, perlu refresh mechanism
3. **Blacklist token** - Saat logout, token harus di-blacklist agar tidak bisa dipakai
4. **Secure transmission** - Gunakan HTTPS (bukan HTTP) saat production
5. **CORS Configuration** - Pastikan CORS dikonfigurasi dengan benar untuk frontend

---

## 🧪 Ringkasan Testing

| # | Endpoint | Method | Protected | Status | Deskripsi |
|----|----------|--------|-----------|--------|-----------|
| 1 | `/api/register` | POST | ❌ | 201 | Register user baru |
| 2 | `/api/login` | POST | ❌ | 200 | Login & dapatkan token |
| 3 | `/api/profile` | GET | ✅ | 200 | Lihat profil user (memerlukan token) |
| 4 | `/api/token-check` | GET | ✅ | 200 | Verifikasi token valid (memerlukan token) |
| 5 | `/api/logout` | POST | ✅ | 200 | Logout & invalidate token (memerlukan token) |
| 6 | `/api/ping` | GET | ❌ | 200 | Health check API |

---

## 📌 Catatan Penting

- **Base URL Development:** `http://localhost:8000`
- **Base URL Production:** Sesuaikan dengan domain production
- **JWT Secret:** Tersimpan di `.env` file (`JWT_SECRET`)
- **JWT Algorithm:** HS256 (HMAC SHA-256)
- **Middleware:** `dummy.jwt` - Custom middleware untuk validasi JWT
- **Expiration:** Token di-set dengan expiration time (default 1 jam)

---

## 🔗 Referensi File

- Controller: `app/Http/Controllers/Api/AuthController.php`
- Middleware: `app/Http/Middleware/DummyJwtMiddleware.php`
- Routes: `routes/api.php`
- Model: `app/Models/DummyUser.php`
- Config: `config/jwt.php`, `config/auth.php`
- Environment: `.env`

---

**Dokumentasi dibuat pada:** April 29, 2026
**Framework:** Laravel Lumen 10.0
**Package JWT:** tymon/jwt-auth v2.1.0
