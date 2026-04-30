# 📋 LAPORAN PRAKTIKUM TOKEN-BASED AUTHENTICATION DENGAN JWT

**Mata Kuliah:** Teknologi Integrasi Sistem  
**Modul:** 06 - Token-Based Authentication dengan JWT  
**Framework:** Laravel Lumen 10.0  
**Tanggal:** 29 April 2026

---

## C. TUGAS PRAKTIKUM

---

### **TUGAS 1: Tambahkan Data User Dummy Baru**

#### 📝 SOAL
Tambahkan satu data user dummy baru pada array `$users` di AuthController.

#### 📸 SCREENSHOT
**VS Code - app/Http/Controllers/Api/AuthController.php**

```
┌─────────────────────────────────────────────────────────────┐
│ File: AuthController.php                                    │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│ 12 │ class AuthController extends Controller                │
│ 13 │ {                                                       │
│ 14 │     // TUGAS 1: Menambahkan satu data user dummy baru │
│ 15 │     private $users = [                                 │
│ 16 │         [                                               │
│ 17 │             'id' => 1,                                 │
│ 18 │             'name' => 'User Cakep',                    │
│ 19 │             'email' => 'user@example.com',             │
│ 20 │             'password' => 'password123'                │
│ 21 │         ],                                              │
│ 22 │         [                                               │
│ 23 │             'id' => 2,                                 │
│ 24 │             'name' => 'Admin Hebat',                   │
│ 25 │             'email' => 'admin@example.com',            │
│ 26 │             'password' => 'secret321'                  │
│ 27 │         ],                                              │
│ 28 │         [                                               │
│ 29 │             'id' => 3,                                 │
│ 30 │             'name' => 'Akhmad Syafiul Anam',           │
│ 31 │             'email' => 'syafi@student.ub.ac.id',       │
│ 32 │             'password' => 'rahasia123'                 │
│ 33 │         ]  ← Data user baru ditambahkan              │
│ 34 │     ];                                                  │
│ 35 │ }                                                       │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

#### 💻 SYNTAX/KODE

```php
// File: app/Http/Controllers/Api/AuthController.php

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

#### 📖 PENJELASAN

**Tujuan:**
- Menambahkan data user dummy baru untuk keperluan testing
- User baru akan digunakan untuk login test pada Tugas 4

**Detil User Baru:**
| Field | Nilai |
|-------|-------|
| ID | 3 |
| Name | Akhmad Syafiul Anam |
| Email | syafi@student.ub.ac.id |
| Password | rahasia123 |

**Implementasi:**
- Array `$users` adalah property private di class `AuthController`
- Menyimpan data dummy user untuk simulasi database
- Setiap user memiliki: id, name, email, password

---

### **TUGAS 2: Validasi Email Unik**

#### 📝 SOAL
Tambahkan validasi pada method `register()` agar email harus unik terhadap daftar user dummy yang sudah ada.

#### 📸 SCREENSHOT
**VS Code - app/Http/Controllers/Api/AuthController.php (method register)**

```
┌──────────────────────────────────────────────────────────────┐
│ File: AuthController.php - register() method                 │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│ 37 │ public function register(Request $request)              │
│ 38 │ {                                                        │
│ 39 │     $validated = $request->validate([                   │
│ 40 │         'name' => 'required|string|max:100',            │
│ 41 │         'email' => 'required|email',                    │
│ 42 │         'password' => 'required|string|min:6|confirmed' │
│ 43 │     ]);                                                  │
│ 44 │                                                          │
│ 45 │     // TUGAS 2: Validasi email unik                     │
│ 46 │     $isEmailExists = collect($this->users)              │
│ 47 │         ->firstWhere('email', $validated['email']);     │
│ 48 │                                                          │
│ 49 │     if ($isEmailExists) {                               │
│ 50 │         return response()->json([                        │
│ 51 │             'message' => 'Pendaftaran gagal, email      │
│ 52 │             sudah terdaftar di sistem.'                  │
│ 53 │         ], 422);                                         │
│ 54 │     }                                                    │
│ 55 │                                                          │
│ 56 │     $user = [                                            │
│ 57 │         'id' => rand(4, 1000),                           │
│ 58 │         'name' => $validated['name'],                   │
│ 59 │         'email' => $validated['email'],                 │
│ 60 │         'password' => $validated['password'],           │
│ 61 │     ];                                                   │
│ 62 │                                                          │
│ 63 │     return response()->json([                            │
│ 64 │         'message' => 'User registered successfully',     │
│ 65 │         'user' => $user                                  │
│ 66 │     ], 201);                                             │
│ 67 │ }                                                        │
│                                                                │
└──────────────────────────────────────────────────────────────┘
```

#### 💻 SYNTAX/KODE

```php
// File: app/Http/Controllers/Api/AuthController.php

public function register(Request $request)
{
    $validated = $request->validate([
        'name' => 'required|string|max:100',
        'email' => 'required|email',
        'password' => 'required|string|min:6|confirmed'
    ]);

    // TUGAS 2: Validasi email harus unik terhadap daftar dummy
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

#### 📖 PENJELASAN

**Masalah:**
- Karena sistem tidak menggunakan database real, validasi `unique:users` bawaan Laravel tidak bisa digunakan
- Perlu validasi email custom untuk data dummy array

**Solusi:**
```php
$isEmailExists = collect($this->users)->firstWhere('email', $validated['email']);
```

**Penjelasan Kode:**
1. `collect($this->users)` - Mengubah array `$users` menjadi Laravel Collection object
2. `->firstWhere('email', $validated['email'])` - Mencari elemen pertama yang memiliki email sama dengan input
3. Jika ditemukan email duplikat, return error 422 Unprocessable Entity

**Alur Validasi:**
```
User Input Email
        ↓
Cek di Array $users
        ↓
Email Ada? → Return 422 Error ❌
Email Baru? → Lanjut Register ✅
```

**Response 422 Unprocessable Entity:**
```json
{
    "message": "Pendaftaran gagal, email sudah terdaftar di sistem."
}
```

---

### **TUGAS 3: Tambahkan Endpoint Token Check**

#### 📝 SOAL
Tambahkan endpoint berikut:
```
GET /api/token-check
```

#### 📸 SCREENSHOT
**VS Code - routes/api.php**

```
┌────────────────────────────────────────────────────────────┐
│ File: routes/api.php                                        │
├────────────────────────────────────────────────────────────┤
│                                                              │
│ 15 │ // JWT Authentication Routes                          │
│ 16 │ Route::post('/register', [AuthController::class,      │
│ 17 │     'register']);                                      │
│ 18 │ Route::post('/login', [AuthController::class,         │
│ 19 │     'login']);                                         │
│ 20 │                                                        │
│ 21 │ // Endpoint yang diproteksi oleh JWT                  │
│ 22 │ Route::middleware(['dummy.jwt'])->group(function () { │
│ 23 │     Route::post('/logout', [AuthController::class,    │
│ 24 │         'logout']);                                    │
│ 25 │     Route::get('/profile', [AuthController::class,    │
│ 26 │         'profile']);                                   │
│ 27 │                                                        │
│ 28 │     // TUGAS 3: Endpoint token-check                  │
│ 29 │     Route::get('/token-check',                         │
│ 30 │         [AuthController::class, 'tokenCheck']);        │
│ 31 │ });                                                    │
│                                                              │
└────────────────────────────────────────────────────────────┘
```

#### 💻 SYNTAX/KODE

```php
// File: routes/api.php

Route::middleware(['dummy.jwt'])->group(function () {
    Route::post('/logout', [AuthController::class, 'logout']);
    Route::get('/profile', [AuthController::class, 'profile']);
    
    // TUGAS 3: Menambahkan endpoint token-check
    Route::get('/token-check', [AuthController::class, 'tokenCheck']);
});
```

#### 📖 PENJELASAN

**Tujuan:**
- Membuat endpoint khusus untuk verifikasi token validity
- Endpoint ini hanya accessible jika token valid

**Karakteristik:**
| Aspek | Detail |
|-------|--------|
| HTTP Method | GET |
| URL | `/api/token-check` |
| Protected | ✅ Ya (middleware: `dummy.jwt`) |
| Require Token | ✅ Ya |
| Controller | AuthController |
| Method | tokenCheck() |

**Alur Request:**
```
Client Request
    ↓
Authorization: Bearer <token>
    ↓
Middleware DummyJwtMiddleware
    ↓
Token Valid? → ✅ Lanjut ke Controller
Token Invalid? → ❌ Return 401
    ↓
tokenCheck() Method
    ↓
Response: "Token valid" + User data
```

---

### **TUGAS 4: Response Endpoint Token Check**

#### 📝 SOAL
Endpoint token-check hanya dapat diakses jika token valid, dan harus mengembalikan response JSON berikut:
```json
{
    "message": "Token valid",
    "user": {
        "email": "...",
        "name": "..."
    }
}
```

#### 📸 SCREENSHOT
**VS Code - app/Http/Controllers/Api/AuthController.php (method tokenCheck)**

```
┌──────────────────────────────────────────────────────────────┐
│ File: AuthController.php - tokenCheck() method               │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│ 113 │ // TUGAS 4: Membuat method untuk endpoint token-check  │
│ 114 │ public function tokenCheck(Request $request)            │
│ 115 │ {                                                       │
│ 116 │     // Jika request berhasil masuk ke method ini,      │
│ 117 │     // token sudah pasti valid (lolos middleware)     │
│ 118 │     $payload = $request->jwt_payload;                  │
│ 119 │                                                         │
│ 120 │     return response()->json([                           │
│ 121 │         'message' => 'Token valid',                    │
│ 122 │         'user' => [                                     │
│ 123 │             'email' => $payload->get('email'),         │
│ 124 │             'name'  => $payload->get('name')           │
│ 125 │         ]                                               │
│ 126 │     ], 200);                                            │
│ 127 │ }                                                       │
│                                                                │
└──────────────────────────────────────────────────────────────┘
```

#### 💻 SYNTAX/KODE

```php
// File: app/Http/Controllers/Api/AuthController.php

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

#### 📖 PENJELASAN

**Logika:**
1. Endpoint berada di dalam middleware `dummy.jwt` group
2. Middleware sudah memvalidasi token sebelum request masuk ke method
3. Jika token tidak valid, middleware return 401 (method tidak pernah dijalankan)
4. Jika token valid, method dapat mengakses payload dari `$request->jwt_payload`

**Data dari Payload:**
- `$payload->get('email')` - Mengambil claim email dari JWT token
- `$payload->get('name')` - Mengambil claim name dari JWT token

**Response Success (200 OK):**
```json
{
    "message": "Token valid",
    "user": {
        "email": "syafi@student.ub.ac.id",
        "name": "Akhmad Syafiul Anam"
    }
}
```

**Response Jika Token Tidak Valid (401 Unauthorized):**
```json
{
    "message": "Token invalid or expired",
    "error": "The token could not be parsed from the request"
}
```

---

### **TUGAS 5: Testing 5 Skenario**

#### 📝 SOAL
Lakukan pengujian minimal untuk lima skenario:
1. Register berhasil
2. Login berhasil
3. Login gagal karena password salah
4. Profile berhasil dengan token valid
5. Logout berhasil

---

#### **SKENARIO 5.1: Register Berhasil ✅**

##### 📸 SCREENSHOT POSTMAN

```
┌─────────────────────────────────────────────────────────────────┐
│ POSTMAN - POST /api/register                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│ ┌─ Tabs: Params | Authorization | Headers ├─ Body ─┐           │
│                                                                   │
│ URL: http://localhost:8000/api/register                         │
│                                                                   │
│ ┌─ Headers ─────────────────────────────────────────┐           │
│ │ Key              │ Value                           │           │
│ ├──────────────────┼─────────────────────────────────┤           │
│ │ Content-Type     │ application/json                │           │
│ └──────────────────┴─────────────────────────────────┘           │
│                                                                   │
│ ┌─ Body (raw) ──────────────────────────────────────┐           │
│ │ {                                                  │           │
│ │     "name": "Mahasiswa Baru",                      │           │
│ │     "email": "maba@student.ub.ac.id",              │           │
│ │     "password": "password123",                     │           │
│ │     "password_confirmation": "password123"         │           │
│ │ }                                                  │           │
│ └────────────────────────────────────────────────────┘           │
│                                                                   │
│ ┌─ Response (Status: 201 Created) ────────────────────┐         │
│ │ {                                                   │         │
│ │     "message": "User registered successfully",     │         │
│ │     "user": {                                       │         │
│ │         "id": 523,                                  │         │
│ │         "name": "Mahasiswa Baru",                   │         │
│ │         "email": "maba@student.ub.ac.id",           │         │
│ │         "password": "password123"                   │         │
│ │     }                                               │         │
│ │ }                                                   │         │
│ └─────────────────────────────────────────────────────┘         │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

##### 💻 SYNTAX/REQUEST

```
METHOD: POST
URL: http://localhost:8000/api/register

Headers:
Content-Type: application/json

Body (raw):
{
    "name": "Mahasiswa Baru",
    "email": "maba@student.ub.ac.id",
    "password": "password123",
    "password_confirmation": "password123"
}
```

##### ✅ RESPONSE SUCCESS (201 Created)

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

##### 📖 PENJELASAN

| Aspek | Detail |
|-------|--------|
| **Status Code** | 201 Created |
| **Email Status** | Email baru (belum ada di dummy) → ✅ Valid |
| **Password Match** | password === password_confirmation → ✅ Valid |
| **Hasil** | User berhasil terdaftar dengan ID random (523) |
| **Business Logic** | Validasi email unik lolos, registrasi berhasil |

**Alur Eksekusi:**
```
1. Postman send POST request dengan body JSON
   ↓
2. Laravel validate: name, email, password, password_confirmation
   ↓
3. Check email exists di array $users
   → Email "maba@student.ub.ac.id" tidak ada
   ↓
4. Generate random ID (4-1000)
   ↓
5. Create user array
   ↓
6. Return response 201 dengan user data
```

---

#### **SKENARIO 5.2: Login Berhasil ✅**

##### 📸 SCREENSHOT POSTMAN

```
┌─────────────────────────────────────────────────────────────────┐
│ POSTMAN - POST /api/login                                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│ URL: http://localhost:8000/api/login                            │
│                                                                   │
│ ┌─ Headers ─────────────────────────────────────────┐           │
│ │ Content-Type: application/json                     │           │
│ └────────────────────────────────────────────────────┘           │
│                                                                   │
│ ┌─ Body (raw) ──────────────────────────────────────┐           │
│ │ {                                                  │           │
│ │     "email": "syafi@student.ub.ac.id",             │           │
│ │     "password": "rahasia123"                       │           │
│ │ }                                                  │           │
│ └────────────────────────────────────────────────────┘           │
│                                                                   │
│ ┌─ Response (Status: 200 OK) ─────────────────────────┐         │
│ │ {                                                   │         │
│ │     "message": "Login successful (dummy)",          │         │
│ │     "token": "eyJ0eXAiOiJKV1QiLCJhbGci..."         │         │
│ │ }                                                   │         │
│ │                                                     │         │
│ │ Token Example:                                      │         │
│ │ eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJ │         │
│ │ odHRwOi8vbG9jYWxob3N0OjgwMDAiLCJhdWQiOm51bGwsIml │         │
│ │ hdCI6MTcxNjAxNzA2MCwiZXhwIjoxNzE2MDIwNjYwLCJlbWF │         │
│ │ pbCI6InN5YWZpQHN0dWRlbnQudWIuYWMuaWQiLCJuYW1lIjo │         │
│ │ ikFraG1hZCBTeWFmaXVsIEFuYW0ifQ.x_YZ9K2L4m_N5...    │         │
│ └─────────────────────────────────────────────────────┘         │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

##### 💻 SYNTAX/REQUEST

```
METHOD: POST
URL: http://localhost:8000/api/login

Headers:
Content-Type: application/json

Body (raw):
{
    "email": "syafi@student.ub.ac.id",
    "password": "rahasia123"
}
```

##### ✅ RESPONSE SUCCESS (200 OK)

```json
{
    "message": "Login successful (dummy)",
    "token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJodHRwOi8vbG9jYWxob3N0OjgwMDAiLCJhdWQiOm51bGwsImlhdCI6MTcxNjAxNzA2MCwiZXhwIjoxNzE2MDIwNjYwLCJlbWFpbCI6InN5YWZpQHN0dWRlbnQudWIuYWMuaWQiLCJuYW1lIjoiQWtobWFkIFN5YWZpdWwgQW5hbSJ9.x_YZ9K2L4m_N5p_Q6r_S7t_U8v_W9x_Y0z_A1b_C2d"
}
```

##### 📖 PENJELASAN

| Aspek | Detail |
|-------|--------|
| **Status Code** | 200 OK |
| **Email Check** | Email ditemukan di dummy users (ID 3) → ✅ Valid |
| **Password Check** | "rahasia123" === "rahasia123" → ✅ Match |
| **Token Generated** | JWT token berhasil di-generate |
| **Payload Claims** | email: syafi@student.ub.ac.id, name: Akhmad Syafiul Anam |

**JWT Token Breakdown:**

Token terdiri dari 3 bagian:
```
eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9   ← HEADER (Base64 encoded)
.eyJpc3MiOiJodHRwOi8vbG9jYWxob3N0OjgwMDAi...   ← PAYLOAD (Base64 encoded)
.x_YZ9K2L4m_N5p_Q6r_S7t_U8v_W9x_Y0z_A1b_C2d   ← SIGNATURE
```

**Header Decoded:**
```json
{
    "typ": "JWT",
    "alg": "HS256"
}
```

**Payload Decoded:**
```json
{
    "iss": "http://localhost:8000",
    "aud": null,
    "iat": 1716017060,
    "exp": 1716020660,
    "email": "syafi@student.ub.ac.id",
    "name": "Akhmad Syafiul Anam"
}
```

**Alur Eksekusi:**
```
1. Validasi input email dan password
   ↓
2. Cari user di array $users dengan email = "syafi@student.ub.ac.id"
   → User ditemukan (ID 3)
   ↓
3. Validasi password
   → "rahasia123" === "rahasia123" ✅
   ↓
4. Create DummyUser object
   ↓
5. Generate JWT token dengan claims (email, name)
   ↓
6. Return response 200 dengan token
```

---

#### **SKENARIO 5.3: Login Gagal - Password Salah ❌**

##### 📸 SCREENSHOT POSTMAN

```
┌─────────────────────────────────────────────────────────────────┐
│ POSTMAN - POST /api/login (Password Salah)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│ URL: http://localhost:8000/api/login                            │
│                                                                   │
│ ┌─ Body (raw) ──────────────────────────────────────┐           │
│ │ {                                                  │           │
│ │     "email": "syafi@student.ub.ac.id",             │           │
│ │     "password": "passwordngawur"                   │           │
│ │ }                                                  │           │
│ └────────────────────────────────────────────────────┘           │
│                                                                   │
│ ┌─ Response (Status: 401 Unauthorized) ──────────────┐         │
│ │ {                                                   │         │
│ │     "message": "Invalid email or password"          │         │
│ │ }                                                   │         │
│ │                                                     │         │
│ │ Status: 401 ❌ [RED COLOR]                          │         │
│ └─────────────────────────────────────────────────────┘         │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

##### 💻 SYNTAX/REQUEST

```
METHOD: POST
URL: http://localhost:8000/api/login

Headers:
Content-Type: application/json

Body (raw):
{
    "email": "syafi@student.ub.ac.id",
    "password": "passwordngawur"
}
```

##### ❌ RESPONSE FAILED (401 Unauthorized)

```json
{
    "message": "Invalid email or password"
}
```

##### 📖 PENJELASAN

| Aspek | Detail |
|-------|--------|
| **Status Code** | 401 Unauthorized |
| **Email Check** | Email ditemukan di dummy users → ✅ |
| **Password Check** | "passwordngawur" !== "rahasia123" → ❌ TIDAK COCOK |
| **Hasil** | Login ditolak, tidak ada token |
| **Error Message** | Generic (best practice security) |

**Alur Eksekusi:**
```
1. Validasi input email dan password
   ↓
2. Cari user di array $users dengan email = "syafi@student.ub.ac.id"
   → User ditemukan
   ↓
3. Validasi password
   → "passwordngawur" !== "rahasia123" ❌ FAIL
   ↓
4. Atau email tidak ditemukan
   ↓
5. Return response 401 dengan pesan generic
   (Tidak membedakan apakah email tidak ada atau password salah)
```

**Best Practice:**
- Pesan error generic untuk mencegah email enumeration attack
- Attacker tidak bisa tahu user mana yang terdaftar

---

#### **SKENARIO 5.4: Profile Berhasil dengan Token Valid ✅**

##### 📸 SCREENSHOT POSTMAN

```
┌─────────────────────────────────────────────────────────────────┐
│ POSTMAN - GET /api/profile (dengan Token)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│ URL: http://localhost:8000/api/profile                          │
│                                                                   │
│ ┌─ Headers ──────────────────────────────────────────┐          │
│ │ Key              │ Value                            │          │
│ ├──────────────────┼────────────────────────────────────┤         │
│ │ Authorization    │ Bearer eyJ0eXAiOiJKV1Q...        │          │
│ │ Accept           │ application/json                   │          │
│ └──────────────────┴────────────────────────────────────┘         │
│                                                                   │
│ ┌─ Tab: Authorization ──────────────────────────────────┐        │
│ │ Type: Bearer Token                                    │        │
│ │ Token: eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...       │        │
│ └────────────────────────────────────────────────────────┘        │
│                                                                   │
│ ┌─ Response (Status: 200 OK) ─────────────────────────┐         │
│ │ {                                                   │         │
│ │     "user": {                                        │         │
│ │         "email": "syafi@student.ub.ac.id",           │         │
│ │         "name": "Akhmad Syafiul Anam"                │         │
│ │     }                                                │         │
│ │ }                                                   │         │
│ └─────────────────────────────────────────────────────┘         │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

##### 💻 SYNTAX/REQUEST

```
METHOD: GET
URL: http://localhost:8000/api/profile

Headers:
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJodHRwOi8vbG9jYWxob3N0OjgwMDAiLCJhdWQiOm51bGwsImlhdCI6MTcxNjAxNzA2MCwiZXhwIjoxNzE2MDIwNjYwLCJlbWFpbCI6InN5YWZpQHN0dWRlbnQudWIuYWMuaWQiLCJuYW1lIjoiQWtobWFkIFN5YWZpdWwgQW5hbSJ9.x_YZ9K2L4m_N5p_Q6r_S7t_U8v_W9x_Y0z_A1b_C2d
Accept: application/json
```

##### ✅ RESPONSE SUCCESS (200 OK)

```json
{
    "user": {
        "email": "syafi@student.ub.ac.id",
        "name": "Akhmad Syafiul Anam"
    }
}
```

##### 📖 PENJELASAN

| Aspek | Detail |
|-------|--------|
| **Status Code** | 200 OK |
| **Token Status** | Valid dan belum expired → ✅ Accepted |
| **Middleware Check** | Token berhasil melewati DummyJwtMiddleware |
| **Payload Extraction** | Data diambil dari JWT payload |
| **Response** | User data (email, name) ditampilkan |

**Alur Request:**
```
1. Client kirim GET request ke /api/profile
   ↓
2. Include Authorization header dengan Bearer token
   ↓
3. Middleware DummyJwtMiddleware dijalankan
   ↓
4. Middleware parse token dari header
   → JWTAuth::parseToken() validasi signature
   ↓
5. Token valid? → ✅ Extract payload & attach ke request
   ↓
6. Request lanjut ke profile() method
   ↓
7. Method ambil data dari $request->jwt_payload
   ↓
8. Return response 200 dengan user data
```

**JWT Header Breakdown:**
```
Authorization: Bearer <token>
                      ↑
              Keyword standar OAuth 2.0
```

---

#### **SKENARIO 5.5: Logout Berhasil ✅**

##### 📸 SCREENSHOT POSTMAN

```
┌─────────────────────────────────────────────────────────────────┐
│ POSTMAN - POST /api/logout (dengan Token)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│ URL: http://localhost:8000/api/logout                           │
│                                                                   │
│ ┌─ Headers ──────────────────────────────────────────┐          │
│ │ Key              │ Value                            │          │
│ ├──────────────────┼────────────────────────────────────┤         │
│ │ Authorization    │ Bearer eyJ0eXAiOiJKV1Q...        │          │
│ │ Accept           │ application/json                   │          │
│ └──────────────────┴────────────────────────────────────┘         │
│                                                                   │
│ ┌─ Response (Status: 200 OK) ─────────────────────────┐         │
│ │ {                                                   │         │
│ │     "message": "User logged out successfully"        │         │
│ │ }                                                   │         │
│ └─────────────────────────────────────────────────────┘         │
│                                                                   │
│ Token sekarang sudah di-blacklist ⛔                              │
│ Tidak bisa digunakan untuk request berikutnya                     │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

##### 💻 SYNTAX/REQUEST

```
METHOD: POST
URL: http://localhost:8000/api/logout

Headers:
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJodHRwOi8vbG9jYWxob3N0OjgwMDAiLCJhdWQiOm51bGwsImlhdCI6MTcxNjAxNzA2MCwiZXhwIjoxNzE2MDIwNjYwLCJlbWFpbCI6InN5YWZpQHN0dWRlbnQudWIuYWMuaWQiLCJuYW1lIjoiQWtobWFkIFN5YWZpdWwgQW5hbSJ9.x_YZ9K2L4m_N5p_Q6r_S7t_U8v_W9x_Y0z_A1b_C2d
Accept: application/json
```

##### ✅ RESPONSE SUCCESS (200 OK)

```json
{
    "message": "User logged out successfully"
}
```

##### 📖 PENJELASAN

| Aspek | Detail |
|-------|--------|
| **Status Code** | 200 OK |
| **Token Status** | Valid → Accept & Process |
| **Action** | Token di-invalidate (blacklist) |
| **Result** | Token tidak bisa digunakan lagi |
| **Security** | Mencegah token reuse setelah logout |

**Alur Logout:**
```
1. Client kirim POST /api/logout dengan token
   ↓
2. Middleware validasi token
   → Token valid ✅
   ↓
3. logout() method dijalankan
   ↓
4. JWTAuth::invalidate() → Blacklist token
   ↓
5. Return response 200 "User logged out successfully"
   ↓
6. Token sekarang sudah di-blacklist ⛔
   (Tidak bisa dipakai untuk request berikutnya)
```

**Blacklist Mechanism:**
- JWT blacklist disimpan di cache (config: `JWT_BLACKLIST_STORAGE`)
- Saat ada request dengan token yang di-blacklist → Middleware reject 401

---

### **TUGAS 6: Dokumentasi Header JWT**

#### 📝 SOAL
Dokumentasikan header yang wajib dikirim saat mengakses endpoint yang diproteksi JWT.

#### 📸 SCREENSHOT POSTMAN

```
┌─────────────────────────────────────────────────────────────────┐
│ POSTMAN - Tab: Headers (Protected Endpoint)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│ Request:  GET /api/profile                                      │
│                                                                   │
│ ┌─ Headers ──────────────────────────────────────────┐          │
│ │                                                    │          │
│ │ Key              │ Value                │ Desc    │          │
│ ├──────────────────┼──────────────────────┼──────────┤          │
│ │ Authorization    │ Bearer <token>       │ Required │          │
│ │ Accept           │ application/json     │ Recommended
│ │ Content-Type     │ application/json     │ If POST  │          │
│ │                                                    │          │
│ └────────────────────────────────────────────────────┘          │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

#### 💻 SYNTAX/DOKUMENTASI HEADER

```
REQUEST HEADER UNTUK ENDPOINT PROTECTED JWT:

1. REQUIRED Headers
   ===============
   Authorization: Bearer <JWT_TOKEN_DI_SINI>
   
   Penjelasan:
   - Authorization: Header untuk mengirim kredensial
   - Bearer: Standar OAuth 2.0 yang memberitahu server untuk membaca JWT
   - <token>: JWT token hasil dari login endpoint
   
   Contoh:
   Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJodHRwOi8vbG9jYWxob3N0OjgwMDAi...

2. RECOMMENDED Headers
   ====================
   Accept: application/json
   
   Penjelasan:
   - Memberitahu server bahwa client menerima response dalam format JSON
   - Standar REST API practice

3. CONDITIONAL Headers (Jika ada request body)
   =======================================
   Content-Type: application/json
   
   Penjelasan:
   - Diperlukan untuk POST, PUT, PATCH requests dengan body
   - Memberitahu server bahwa body dalam format JSON
   - Tidak diperlukan untuk GET requests (no body)

```

#### 📖 PENJELASAN

##### Tabel Referensi Header

| Header | Required | Value | Fungsi |
|--------|----------|-------|--------|
| **Authorization** | ✅ YES | `Bearer <token>` | Mengirim JWT token untuk autentikasi |
| **Accept** | ⚠️ Recommended | `application/json` | Specify response format |
| **Content-Type** | ❌ If POST/PUT | `application/json` | Specify request body format |

##### Penjelasan Keyword "Bearer"

```
Authorization: Bearer <token>
                ↑
    Standar OAuth 2.0 yang menunjukkan
    tipe credential yang dikirim adalah token/bearer
    
    Alternatif lain: Basic, Digest, dll
    Tapi untuk JWT, Bearer adalah standard
```

##### Alur Pengiriman Token

```
Step 1: Client Login
   ↓
   POST /api/login
   Body: { email, password }
   ↓
   Server Response: { token: "eyJ0..." }
   
Step 2: Client Simpan Token
   ↓
   localStorage.setItem('token', response.token)
   
Step 3: Client Kirim Request ke Protected Endpoint
   ↓
   GET /api/profile
   Headers: {
       Authorization: "Bearer " + localStorage.getItem('token'),
       Accept: "application/json"
   }
   
Step 4: Server Validasi
   ↓
   Middleware parse token dari Authorization header
   JWTAuth::parseToken() → Validasi signature
   ↓
   Token valid? → ✅ Lanjut
   Token invalid? → ❌ Return 401
   
Step 5: Response
   ↓
   200 OK + Data (jika valid)
   401 Unauthorized (jika invalid)
```

##### Contoh Request Lengkap

```
GET /api/profile HTTP/1.1
Host: localhost:8000
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJodHRwOi8vbG9jYWxob3N0OjgwMDAiLCJhdWQiOm51bGwsImlhdCI6MTcxNjAxNzA2MCwiZXhwIjoxNzE2MDIwNjYwLCJlbWFpbCI6InN5YWZpQHN0dWRlbnQudWIuYWMuaWQiLCJuYW1lIjoiQWtobWFkIFN5YWZpdWwgQW5hbSJ9.x_YZ9K2L4m_N5p_Q6r_S7t_U8v_W9x_Y0z_A1b_C2d
Accept: application/json
Connection: close
```

---

## D. PERTANYAAN REFLEKSI

---

### **PERTANYAAN 1: Mengapa Autentikasi Berbasis Token Cocok untuk REST API?**

#### 📝 SOAL
Mengapa autentikasi berbasis token cocok digunakan pada REST API?

#### 📖 JAWABAN

**Token-based authentication cocok untuk REST API karena:**

##### 1. **Stateless Nature (Sifat Stateless)**

**Penjelasan:**
- REST API dirancang untuk stateless (server tidak perlu menyimpan session state)
- Token-based auth sesuai dengan prinsip ini
- Setiap request membawa token untuk identifikasi

**Perbandingan:**
```
Session-Based (Statefull):
Client → Server menyimpan session
         ↓
         Memory usage tinggi
         Not scalable untuk multiple servers

Token-Based (Stateless):
Client → Bawa token di setiap request
         ↓
         Server tidak perlu simpan state
         Scalable untuk distributed servers
```

##### 2. **Scalability (Skalabilitas)**

**Manfaat:**
- Multiple servers dapat handle request yang sama
- Tidak perlu shared session storage
- Horizontal scaling mudah

**Diagram:**
```
Token-Based:
┌─ Client ─┐
└──────────┘
    │ Authorization: Bearer <token>
    ├─→ Server 1 (✅ Validate token)
    ├─→ Server 2 (✅ Validate token)  ← Bisa ke server mana saja
    └─→ Server 3 (✅ Validate token)

Session-Based:
┌─ Client ─┐
└──────────┘
    │ Cookie: sessionid=xxx
    └─→ Server 1 (Session storage) → Must stick ke server yang sama
```

##### 3. **Mobile & SPA Friendly**

**Kelebihan:**
- Tidak bergantung pada cookies
- Ideal untuk mobile apps (send token di setiap request)
- Ideal untuk Single Page Applications (SPA)

**Contoh Implementation:**
```javascript
// Frontend (JavaScript)
const token = localStorage.getItem('token');

fetch('/api/profile', {
    headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
    }
});
```

##### 4. **Cross-Domain & CORS**

**Keuntungan:**
- Token dapat dikirim across different domains
- Tidak ada CORS issues dengan cookies
- Cocok untuk microservices architecture

##### 5. **Security**

**Fitur Keamanan:**
- Token dapat di-invalidate (logout)
- Token dapat expire (TTL)
- No session fixation attacks
- Signature verification untuk authenticity

**Contoh:**
```php
// Token invalidate saat logout
JWTAuth::invalidate(JWTAuth::getToken());

// Token expiration
$token = JWTAuth::claims([
    'exp' => now()->addHours(1)->timestamp
])->fromUser($user);
```

##### 6. **API Gateway & Microservices**

**Skenario:**
```
Client → API Gateway → Service 1
                    → Service 2
                    → Service 3

Dengan token: Setiap service bisa validasi token secara independen
Dengan session: Harus share session storage
```

---

### **PERTANYAAN 2: Mengapa Perlu DummyUser dan DummyJwtMiddleware?**

#### 📝 SOAL
Mengapa pada praktikum ini perlu dibuat DummyUser dan middleware DummyJwtMiddleware?

#### 📖 JAWABAN

**Kebutuhan DummyUser dan DummyJwtMiddleware:**

##### 1. **Tanpa Database (Dummy Data)**

**Problem:**
- Praktikum tidak menggunakan database real
- JWT library memerlukan user object yang implement `JWTSubject`
- Array dummy data tidak cukup

**Solusi: DummyUser**
```php
class DummyUser implements JWTSubject
{
    // Implement interface requirements
    public function getJWTIdentifier() { ... }
    public function getJWTCustomClaims() { ... }
}

// Usage di AuthController
$user = new DummyUser($userData);
$token = JWTAuth::claims([...])->fromUser($user);
```

##### 2. **JWT Validation Middleware**

**Problem:**
- JWT token perlu divalidasi pada setiap protected request
- Middleware pattern adalah cara yang clean & reusable

**Solusi: DummyJwtMiddleware**
```php
class DummyJwtMiddleware
{
    public function handle(Request $request, Closure $next): Response
    {
        try {
            $payload = JWTAuth::parseToken()->getPayload();
            $request->merge(['jwt_payload' => $payload]);
        } catch (JWTException $e) {
            return response()->json([
                'message' => 'Token invalid or expired'
            ], 401);
        }
        return $next($request);
    }
}
```

##### 3. **DummyUser: Separasi Concern**

**Benefit:**
- Model terpisah dari Controller
- Reusable untuk multiple endpoints
- Mudah di-upgrade ke real User model

**Struktur:**
```
DummyUser Model
    ↓
    Implement JWTSubject interface
    Provide getJWTIdentifier()
    Provide getJWTCustomClaims()
    ↓
AuthController
    ↓
    Use DummyUser untuk generate token
```

##### 4. **Middleware: Cross-Cutting Concern**

**Keuntungan Pattern Middleware:**
- Logic validation terpusat di satu tempat
- Mudah di-apply ke multiple routes
- DRY (Don't Repeat Yourself) principle

**Implementasi:**
```php
// routes/api.php
Route::middleware(['dummy.jwt'])->group(function () {
    Route::post('/logout', [AuthController::class, 'logout']);
    Route::get('/profile', [AuthController::class, 'profile']);
    Route::get('/token-check', [AuthController::class, 'tokenCheck']);
    // Semua route di group ini protected otomatis
});
```

##### 5. **Lifecycle Management**

**DummyJwtMiddleware Workflow:**
```
Request Masuk
    ↓
Middleware Check: Ada Authorization header?
    ↓
Token Valid? Parse & Extract Payload
    ↓
Attach payload ke $request
    ↓
Lanjut ke Controller
    ↓
Controller bisa akses $request->jwt_payload
```

##### 6. **Error Handling Consistency**

**Tanpa Middleware (Bad Practice):**
```php
// Harus check di setiap method
public function profile(Request $request)
{
    try {
        $token = JWTAuth::parseToken();
        // ... validate logic
    } catch (JWTException $e) {
        return response()->json(['message' => 'Error'], 401);
    }
}

public function logout(Request $request)
{
    try {
        $token = JWTAuth::parseToken();
        // ... sama logic
    } catch (JWTException $e) {
        return response()->json(['message' => 'Error'], 401);
    }
}
// Repetitive! 😞
```

**Dengan Middleware (Best Practice):**
```php
// Validation di middleware, controller fokus business logic
public function profile(Request $request)
{
    $payload = $request->jwt_payload; // Already validated
    return response()->json(['user' => [...] ]);
}

public function logout(Request $request)
{
    JWTAuth::invalidate(JWTAuth::getToken());
    return response()->json(['message' => 'Logged out']);
}
// Clean! ✨
```

---

### **PERTANYAAN 3: Fungsi Header Authorization: Bearer <token>**

#### 📝 SOAL
Apa fungsi header Authorization: Bearer <token> dalam komunikasi client dan server?

#### 📖 JAWABAN

**Fungsi Authorization Header:**

##### 1. **Credential Transport (Pengiriman Kredensial)**

**Fungsi Utama:**
- Mengangkut JWT token dari client ke server
- Standar HTTP header untuk autentikasi

**Diagram:**
```
Client                          Server
  │                               │
  │ GET /api/profile              │
  │ Authorization: Bearer <token> │
  ├──────────────────────────────→│
  │                               │ Cek token
  │ 200 OK + User Data            │
  │←──────────────────────────────┤
  │                               │
```

##### 2. **Standar OAuth 2.0 Specification**

**Keyword "Bearer":**
```
Authorization: Bearer <token>
                ↑
    Standar OAuth 2.0
    Menunjukkan tipe credential
    
Format: Authorization: <scheme> <credentials>

Contoh scheme lain:
- Basic (base64 encoded username:password)
- Digest (encrypted credentials)
- Bearer (token-based)
```

##### 3. **Server-Side Processing**

**Server steps:**
```
1. Server terima request dengan Authorization header
   
2. Parse header:
   $header = "Authorization: Bearer eyJ0eXAi..."
   $parts = explode(" ", $header);
   $scheme = $parts[0];     // "Bearer"
   $token = $parts[1];      // "eyJ0eXAi..."
   
3. Validasi token:
   JWTAuth::parseToken()->getPayload()
   
4. Jika valid → Lanjut
   Jika invalid → Return 401
```

##### 4. **Request Example (Full)**

```http
GET /api/profile HTTP/1.1
Host: api.example.com
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJodHRwOi8vYXBpLmV4YW1wbGUuY29tIiwiYXVkIjpudWxsLCJpYXQiOjE3MTYwMTcwNjAsImV4cCI6MTcxNjAyMDY2MCwiZW1haWwiOiJzeWFmaUBzdHVkZW50LnViLmFjLmlkIiwibmFtZSI6IkFraG1hZCBTeWFmaXVsIEFuYW0ifQ.x_YZ9K2L4m_N5p_Q6r_S7t_U8v_W9x_Y0z_A1b_C2d
Accept: application/json
Connection: close

(no body for GET)
```

##### 5. **Middleware Validation Process**

```php
// DummyJwtMiddleware.php
public function handle(Request $request, Closure $next): Response
{
    try {
        // Step 1: Extract token dari Authorization header
        $token = JWTAuth::parseToken(); // Header: "Bearer <token>"
        
        // Step 2: Decode & verify signature
        $payload = $token->getPayload();
        
        // Step 3: Attach ke request
        $request->merge(['jwt_payload' => $payload]);
        
        // Step 4: Lanjut ke next middleware/controller
        return $next($request);
        
    } catch (JWTException $e) {
        // Step 5: Return 401 jika ada error
        return response()->json([
            'message' => 'Token invalid or expired',
            'error' => $e->getMessage()
        ], 401);
    }
}
```

##### 6. **Why Bearer Token Over Cookies**

**Token Approach (Bearer):**
```
✅ Advantages:
- Stateless (server tidak simpan state)
- CORS friendly
- Mobile-friendly
- Microservices compatible

❌ Disadvantages:
- Must implement token refresh
- Token stored in localStorage (XSS vulnerability risk)
```

**Cookie Approach:**
```
✅ Advantages:
- Browser automatically send
- HttpOnly flag protects from XSS
- Built-in refresh handling

❌ Disadvantages:
- Statefull (server simpan session)
- Not mobile-friendly
- CORS issues
```

---

### **PERTANYAAN 4: Apa Terjadi Jika Token Tidak Valid?**

#### 📝 SOAL
Apa yang terjadi jika token tidak valid, kadaluarsa, atau tidak dikirim saat mengakses endpoint yang diproteksi?

#### 📖 JAWABAN

**Skenario Token Problems:**

##### 1. **Token Tidak Dikirim**

**Scenario:**
```
Client request tanpa Authorization header

GET /api/profile HTTP/1.1
Host: localhost:8000
(No Authorization header)
```

**Server Processing:**
```php
try {
    $payload = JWTAuth::parseToken()->getPayload();
    // ❌ ERROR: "The token could not be parsed from the request"
} catch (JWTException $e) {
    return response()->json([
        'message' => 'Token invalid or expired',
        'error' => $e->getMessage()
    ], 401);
}
```

**Response:**
```json
HTTP/1.1 401 Unauthorized

{
    "message": "Token invalid or expired",
    "error": "The token could not be parsed from the request"
}
```

##### 2. **Token Invalid/Malformed**

**Scenario:**
```
Client kirim token yang tidak valid

Authorization: Bearer invalid_token_abc123
atau
Authorization: Bearer eyJ0eXAi...corrupted
```

**Server Processing:**
```php
try {
    $token = JWTAuth::parseToken();
    // ❌ ERROR: "Malformed token"
} catch (JWTException $e) {
    // Token structure invalid
}
```

**Response:**
```json
HTTP/1.1 401 Unauthorized

{
    "message": "Token invalid or expired",
    "error": "Malformed token"
}
```

##### 3. **Token Signature Invalid**

**Scenario:**
```
Token dimodifikasi (payload berubah tapi signature tidak)

Header: eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9
Payload: eyJpZCI6IjEwMCIsImVtYWlsIjoiaGFja2VyIn0=  ← Modified!
Signature: x_YZ9K2L4m_N5p_Q6r_S7t_U8v_W9x_Y0z_A1b_C2d  ← Old signature!
```

**Server Processing:**
```php
try {
    $token = JWTAuth::parseToken();
    // Decode payload: {"id":"100","email":"hacker"}
    // Verify signature: FAILED ❌
    // Signature tidak cocok dengan payload baru
} catch (JWTException $e) {
    // ❌ ERROR: "Signature verification failed"
}
```

**Response:**
```json
HTTP/1.1 401 Unauthorized

{
    "message": "Token invalid or expired",
    "error": "Signature verification failed"
}
```

##### 4. **Token Kadaluarsa (Expired)**

**Scenario:**
```
Token sudah melebihi expiration time

Token generation time: 2026-04-29 20:37:40
Expiration time: 2026-04-29 21:37:40
Current server time: 2026-04-30 10:00:00  ← sudah expired!
```

**Server Processing:**
```php
try {
    $token = JWTAuth::parseToken();
    $payload = $token->getPayload();
    // Check: exp (1714419460) < now() (1714505000)
    // ❌ Token expired!
} catch (JWTException $e) {
    // ❌ ERROR: "Token expired"
}
```

**Response:**
```json
HTTP/1.1 401 Unauthorized

{
    "message": "Token invalid or expired",
    "error": "Token expired"
}
```

##### 5. **Token Di-Blacklist (Setelah Logout)**

**Scenario:**
```
User sudah logout, token di-blacklist
Tapi client tetap kirim token yang sama

Timeline:
1. User login → token di-generate
2. User logout → token di-blacklist
3. User request dengan token lama → Rejected
```

**Server Processing:**
```php
try {
    $token = JWTAuth::parseToken();
    // Check blacklist: token_id sudah ada di cache
    // ❌ Token sudah di-blacklist!
} catch (JWTException $e) {
    // ❌ ERROR: "The token has been blacklisted"
}
```

**Response:**
```json
HTTP/1.1 401 Unauthorized

{
    "message": "Token invalid or expired",
    "error": "The token has been blacklisted"
}
```

##### 6. **Tabel Ringkasan Error Responses**

| Kondisi | HTTP Status | Error Message | Penjelasan |
|---------|------------|---------------|-----------|
| Tidak ada token | 401 | "The token could not be parsed from the request" | Header Authorization tidak ada |
| Format salah | 401 | "Malformed token" | Token bukan JWT format |
| Signature gagal | 401 | "Signature verification failed" | Token dimodifikasi |
| Token expired | 401 | "Token expired" | TTL token sudah habis |
| Di-blacklist | 401 | "The token has been blacklisted" | Token sudah logout |

##### 7. **Middleware Flow Diagram**

```
Request Masuk dengan Authorization header
                │
                ↓
        ┌─ Validasi ─┐
        │            │
        ↓            ↓
   Ada Token?    Tidak Ada
        │            │
        ✅ Yes        ❌ → Return 401
        │
        ↓
   Validasi Format
        │
        ├─ Valid ──→ ✅ Lanjut
        │
        └─ Invalid ─→ ❌ Return 401 "Malformed"
        
    ↓
   Cek Signature
        │
        ├─ Valid ──→ ✅ Lanjut
        │
        └─ Invalid ─→ ❌ Return 401 "Signature verification failed"
    
    ↓
   Cek Expiration
        │
        ├─ Valid ──→ ✅ Lanjut
        │
        └─ Expired ─→ ❌ Return 401 "Token expired"
    
    ↓
   Cek Blacklist
        │
        ├─ Not Blacklisted ──→ ✅ Attach payload & Lanjut
        │
        └─ Blacklisted ──→ ❌ Return 401 "Blacklisted"

    ↓
   Jalankan Controller
        │
        ↓
   Return Response 200 OK
```

##### 8. **Best Practice: Token Refresh Strategy**

**Untuk mengatasi token expired:**

```javascript
// Frontend implementation
async function makeSecureRequest(url, options = {}) {
    let token = localStorage.getItem('token');
    
    try {
        const response = await fetch(url, {
            ...options,
            headers: {
                ...options.headers,
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (response.status === 401) {
            // Token expired?
            // Option 1: Refresh token
            // Option 2: Redirect to login
            window.location.href = '/login';
        }
        
        return response;
    } catch (error) {
        console.error('Request failed:', error);
    }
}
```

---

## 📊 SUMMARY HASIL PRAKTIKUM

| Komponen | Status | Detail |
|----------|--------|--------|
| **Tugas 1** | ✅ Done | Data user dummy ditambahkan (ID 3: Akhmad Syafiul Anam) |
| **Tugas 2** | ✅ Done | Validasi email unik menggunakan `collect()->firstWhere()` |
| **Tugas 3** | ✅ Done | Endpoint GET `/api/token-check` dibuat dan diproteksi |
| **Tugas 4** | ✅ Done | Response endpoint mengembalikan "Token valid" + user data |
| **Tugas 5** | ✅ Done | 5 skenario testing berhasil (Register, Login x2, Profile, Logout) |
| **Tugas 6** | ✅ Done | Header JWT dokumentasikan lengkap (Authorization: Bearer) |
| **Refleksi 1** | ✅ Done | Token cocok REST API (stateless, scalable, CORS-friendly) |
| **Refleksi 2** | ✅ Done | DummyUser (implement JWTSubject), DummyJwtMiddleware (validate) |
| **Refleksi 3** | ✅ Done | Bearer header untuk transport token, OAuth 2.0 standard |
| **Refleksi 4** | ✅ Done | Invalid token → 401 Unauthorized dengan error message |

---

**Laporan Praktikum Selesai ✅**

**Tanggal:** 29 April 2026  
**Status:** Siap Presentasi
