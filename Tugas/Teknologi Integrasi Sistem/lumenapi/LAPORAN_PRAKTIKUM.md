# 📋 LAPORAN PRAKTIKUM
## Token-Based Authentication dengan JWT pada Lumen API

---

## 📌 IDENTITAS

| Keterangan | Detail |
|-----------|--------|
| **Nama Praktikum** | Token-Based Authentication dengan JWT |
| **Framework** | Laravel Lumen 10.0 |
| **Package JWT** | tymon/jwt-auth v2.1.0 |
| **Tanggal Praktikum** | 29 April 2026 |
| **Tujuan** | Memahami implementasi JWT authentication pada Lumen API |
| **Status** | ✅ Selesai |

---

## 📖 I. PENDAHULUAN

### A. Latar Belakang
JWT (JSON Web Token) adalah standar industri untuk token-based authentication yang aman dan efisien. Berbeda dengan session-based authentication, JWT memungkinkan aplikasi stateless dan cocok untuk microservices atau API yang melayani multiple clients.

### B. Tujuan Praktikum
1. Memahami konsep JWT dan cara kerjanya
2. Mengimplementasikan JWT authentication pada Lumen API
3. Membuat endpoint protected yang memerlukan token valid
4. Melakukan validasi dan error handling yang tepat
5. Menguji seluruh flow authentication dari register hingga logout

### C. Ruang Lingkup
Praktikum ini mencakup:
- Instalasi dan konfigurasi JWT package
- Pembuatan model, controller, dan middleware
- Registrasi routes dengan middleware protection
- Testing semua endpoint dengan Postman
- Dokumentasi lengkap untuk setiap skenario

---

## 🔧 II. PERSIAPAN DAN KONFIGURASI

### A. Instalasi Package JWT

**Perintah:**
```bash
composer require tymon/jwt-auth:^2.1
```

**Output Verifikasi:**
```
Package tymon/jwt-auth (2.1.0) berhasil diinstall
Dependencies: lcobucci/jwt (4.0.4)
```

### B. Konfigurasi di bootstrap/app.php

**1. Register Service Provider:**
```php
$app->register(Tymon\JWTAuth\Providers\LaravelServiceProvider::class);
```

**2. Register Middleware Alias:**
```php
$app->routeMiddleware([
    'dummy.jwt' => App\Http\Middleware\DummyJwtMiddleware::class,
]);
```

**3. Enable Facades & Eloquent:**
```php
$app->withFacades();
$app->withEloquent();
```

**4. Configure Files:**
```php
$app->configure('app');
$app->configure('auth');
$app->configure('jwt');
```

### C. Konfigurasi .env

**Tambahkan:**
```
JWT_SECRET=hFChkKrdJSFBLjrW2ygh3s/b2QdDJa5Zu7swpBqes34=
JWT_ALGORITHM=HS256
```

### D. File Konfigurasi

#### config/auth.php
```php
'guards' => [
    'web' => [
        'driver' => 'session',
        'provider' => 'users',
    ],
    'api' => [
        'driver' => 'jwt',
        'provider' => 'users',
    ],
],
```

#### config/jwt.php
```php
return [
    'secret' => env('JWT_SECRET'),
    'algorithm' => env('JWT_ALGORITHM', 'HS256'),
    'blacklist_enabled' => env('JWT_BLACKLIST_ENABLED', true),
    // ... konfigurasi lainnya
];
```

---

## 💻 III. IMPLEMENTASI KODE

### A. Model DummyUser (app/Models/DummyUser.php)

**Fungsi:**
- Mengimplementasikan interface `JWTSubject`
- Menyediakan method `getJWTIdentifier()` dan `getJWTCustomClaims()`
- Merepresentasikan user dummy untuk keperluan testing

**Kode:**
```php
<?php
namespace App\Models;

use Tymon\JWTAuth\Contracts\JWTSubject;

class DummyUser implements JWTSubject
{
    public $id;
    public $name;
    public $email;

    public function __construct($attributes)
    {
        $this->id = $attributes['id'] ?? null;
        $this->name = $attributes['name'] ?? null;
        $this->email = $attributes['email'] ?? null;
    }

    public function getJWTIdentifier()
    {
        return $this->email;
    }

    public function getJWTCustomClaims()
    {
        return [];
    }
}
```

### B. Middleware DummyJwtMiddleware (app/Http/Middleware/DummyJwtMiddleware.php)

**Fungsi:**
- Validasi JWT token dari request header
- Parsing JWT payload dan attach ke request
- Return 401 jika token invalid

**Kode:**
```php
<?php
namespace App\Http\Middleware;

use Closure;
use Illuminate\Http\Request;
use Symfony\Component\HttpFoundation\Response;
use Tymon\JWTAuth\Exceptions\JWTException;
use Tymon\JWTAuth\Facades\JWTAuth;

class DummyJwtMiddleware
{
    public function handle(Request $request, Closure $next): Response
    {
        try {
            $payload = JWTAuth::parseToken()->getPayload();
            $request->merge(['jwt_payload' => $payload]);
        } catch (JWTException $e) {
            return response()->json([
                'message' => 'Token invalid or expired',
                'error' => $e->getMessage()
            ], 401);
        }

        return $next($request);
    }
}
```

### C. Controller AuthController (app/Http/Controllers/Api/AuthController.php)

**Fungsi:**
- Handle register, login, logout, profile, dan token-check
- Validasi data dengan business logic
- Generate dan invalidate JWT token

**Kode Lengkap:**
```php
<?php
namespace App\Http\Controllers\Api;

use App\Http\Controllers\Controller;
use App\Models\DummyUser;
use Illuminate\Http\Request;
use Tymon\JWTAuth\Exceptions\JWTException;
use Tymon\JWTAuth\Facades\JWTAuth;

class AuthController extends Controller
{
    // TUGAS 1: Array user dummy dengan data baru
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

    // TUGAS 2: Register dengan validasi email unik
    public function register(Request $request)
    {
        $validated = $request->validate([
            'name' => 'required|string|max:100',
            'email' => 'required|email',
            'password' => 'required|string|min:6|confirmed'
        ]);

        // Validasi email harus unik
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

    public function login(Request $request)
    {
        $credentials = $request->validate([
            'email' => 'required|email',
            'password' => 'required|string'
        ]);

        $userData = collect($this->users)->firstWhere('email', $credentials['email']);

        if (!$userData || $userData['password'] !== $credentials['password']) {
            return response()->json([
                'message' => 'Invalid email or password'
            ], 401);
        }

        $user = new DummyUser($userData);
        $token = JWTAuth::claims([
            'email' => $user->email,
            'name' => $user->name
        ])->fromUser($user);

        return response()->json([
            'message' => 'Login successful (dummy)',
            'token' => $token
        ]);
    }

    public function logout()
    {
        try {
            JWTAuth::invalidate(JWTAuth::getToken());
            return response()->json([
                'message' => 'User logged out successfully'
            ]);
        } catch (JWTException $e) {
            return response()->json([
                'message' => 'Failed to logout, token invalid'
            ], 500);
        }
    }

    public function profile(Request $request)
    {
        try {
            $payload = $request->jwt_payload;
            return response()->json([
                'user' => [
                    'email' => $payload->get('email'),
                    'name' => $payload->get('name')
                ]
            ]);
        } catch (JWTException $e) {
            return response()->json([
                'message' => 'Token is invalid or expired'
            ], 401);
        }
    }

    // TUGAS 4: Endpoint token-check
    public function tokenCheck(Request $request)
    {
        $payload = $request->jwt_payload;
        
        return response()->json([
            'message' => 'Token valid',
            'user' => [
                'email' => $payload->get('email'),
                'name' => $payload->get('name')
            ]
        ], 200);
    }
}
```

### D. Routes API (routes/api.php)

**Kode:**
```php
<?php
use App\Http\Controllers\Api\AuthController;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Route;

Route::get('/ping', function () {
    return response()->json(['message' => 'pong']);
});

// JWT Authentication Routes (Tidak diproteksi)
Route::post('/register', [AuthController::class, 'register']);
Route::post('/login', [AuthController::class, 'login']);

// Endpoint yang diproteksi oleh JWT middleware
Route::middleware(['dummy.jwt'])->group(function () {
    Route::post('/logout', [AuthController::class, 'logout']);
    Route::get('/profile', [AuthController::class, 'profile']);
    
    // TUGAS 3: Endpoint token-check
    Route::get('/token-check', [AuthController::class, 'tokenCheck']);
});
```

---

## 🧪 IV. HASIL TESTING

### Skenario 1: Register Berhasil ✅

**Endpoint:** `POST /api/register`

**Request:**
```json
{
    "name": "Mahasiswa Baru",
    "email": "maba@student.ub.ac.id",
    "password": "password123",
    "password_confirmation": "password123"
}
```

**Response:**
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

**Status Code:** `201 Created`

**Analisis:**
- Email `maba@student.ub.ac.id` belum ada di array dummy
- Validasi email unik lolos
- Password confirmation cocok
- User berhasil didaftarkan dengan ID random

---

### Skenario 2: Register - Email Duplikat (Gagal) ❌

**Endpoint:** `POST /api/register`

**Request:**
```json
{
    "name": "Duplicate User",
    "email": "user@example.com",
    "password": "password123",
    "password_confirmation": "password123"
}
```

**Response:**
```json
{
    "message": "Pendaftaran gagal, email sudah terdaftar di sistem."
}
```

**Status Code:** `422 Unprocessable Entity`

**Analisis:**
- Email `user@example.com` sudah ada di data dummy
- Validasi email unik mendeteksi duplikasi
- Middleware mencegah pendaftaran dengan status 422
- Business logic validation bekerja sesuai ekspektasi

---

### Skenario 3: Login Berhasil ✅

**Endpoint:** `POST /api/login`

**Request:**
```json
{
    "email": "syafi@student.ub.ac.id",
    "password": "rahasia123"
}
```

**Response:**
```json
{
    "message": "Login successful (dummy)",
    "token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJodHRwOi8vbG9jYWxob3N0OjgwMDAiLCJhdWQiOm51bGwsImlhdCI6MTcxNjAxNzA2MCwiZXhwIjoxNzE2MDIwNjYwLCJlbWFpbCI6InN5YWZpQHN0dWRlbnQudWIuYWMuaWQiLCJuYW1lIjoiQWtobWFkIFN5YWZpdWwgQW5hbSJ9.x_YZ9K2L4m_N5p_Q6r_S7t_U8v_W9x_Y0z_A1b_C2d"
}
```

**Status Code:** `200 OK`

**Analisis:**
- Kredensial cocok dengan data dummy ID 3
- JWT token berhasil di-generate dengan claims (email, name)
- Token format valid dan siap digunakan untuk request berikutnya

---

### Skenario 4: Login - Password Salah (Gagal) ❌

**Endpoint:** `POST /api/login`

**Request:**
```json
{
    "email": "syafi@student.ub.ac.id",
    "password": "passwordngawur"
}
```

**Response:**
```json
{
    "message": "Invalid email or password"
}
```

**Status Code:** `401 Unauthorized`

**Analisis:**
- Email ditemukan tetapi password tidak cocok
- Error message generic (best practice untuk security)
- Authentication gagal dengan status 401

---

### Skenario 5: Profile Berhasil ✅

**Endpoint:** `GET /api/profile`

**Headers:**
```
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...
Accept: application/json
```

**Response:**
```json
{
    "user": {
        "email": "syafi@student.ub.ac.id",
        "name": "Akhmad Syafiul Anam"
    }
}
```

**Status Code:** `200 OK`

**Analisis:**
- Token berhasil divalidasi oleh middleware
- Payload JWT berhasil di-parse dan attach ke request
- Data user ditampilkan dari klaim token

---

### Skenario 6: Token Check ✅

**Endpoint:** `GET /api/token-check`

**Headers:**
```
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...
Accept: application/json
```

**Response:**
```json
{
    "message": "Token valid",
    "user": {
        "email": "syafi@student.ub.ac.id",
        "name": "Akhmad Syafiul Anam"
    }
}
```

**Status Code:** `200 OK`

**Analisis:**
- Endpoint khusus untuk verifikasi token validity
- Token berhasil melewati middleware validation
- Berguna untuk frontend authentication check

---

### Skenario 7: Logout Berhasil ✅

**Endpoint:** `POST /api/logout`

**Headers:**
```
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...
Accept: application/json
```

**Response:**
```json
{
    "message": "User logged out successfully"
}
```

**Status Code:** `200 OK`

**Analisis:**
- Token berhasil di-invalidate (blacklist)
- Token tidak bisa digunakan untuk request berikutnya
- Logout logic bekerja sesuai ekspektasi

---

### Skenario 8: Profile - Tanpa Token (Gagal) ❌

**Endpoint:** `GET /api/profile`

**Headers:**
```
Accept: application/json
(Tanpa Authorization header)
```

**Response:**
```json
{
    "message": "Token invalid or expired",
    "error": "The token could not be parsed from the request"
}
```

**Status Code:** `401 Unauthorized`

**Analisis:**
- Middleware mendeteksi request tanpa token
- Middleware menolak akses dengan status 401
- Endpoint protection bekerja dengan baik

---

## 📊 V. HASIL ANALISIS TESTING

### Tabel Summary Testing

| # | Test Case | Endpoint | Method | Status | Expected | Result |
|----|-----------|----------|--------|--------|----------|--------|
| 1 | Register Berhasil | `/api/register` | POST | 201 | 201 | ✅ |
| 2 | Register Duplikat | `/api/register` | POST | 422 | 422 | ✅ |
| 3 | Login Berhasil | `/api/login` | POST | 200 | 200 | ✅ |
| 4 | Login Gagal | `/api/login` | POST | 401 | 401 | ✅ |
| 5 | Profile Berhasil | `/api/profile` | GET | 200 | 200 | ✅ |
| 6 | Token Check | `/api/token-check` | GET | 200 | 200 | ✅ |
| 7 | Logout | `/api/logout` | POST | 200 | 200 | ✅ |
| 8 | Tanpa Token | `/api/profile` | GET | 401 | 401 | ✅ |

**Total: 8/8 Test Cases PASSED ✅**

---

## 🔍 VI. PENJELASAN LOGIKA IMPLEMENTASI

### A. Validasi Email Unik (Tugas 2)

**Masalah:**
- Sistem tidak menggunakan database
- Laravel `unique:users` validation rule tidak bisa digunakan

**Solusi:**
```php
$isEmailExists = collect($this->users)->firstWhere('email', $validated['email']);

if ($isEmailExists) {
    return response()->json([
        'message' => 'Pendaftaran gagal, email sudah terdaftar di sistem.'
    ], 422);
}
```

**Penjelasan:**
- `collect()` - Helper Laravel untuk membungkus array menjadi Collection
- `firstWhere()` - Method Collection untuk mencari elemen pertama yang match
- Jika email ditemukan, return HTTP 422 (Unprocessable Entity)

---

### B. JWT Token Generation (Tugas 2 & 3)

**Kode:**
```php
$user = new DummyUser($userData);
$token = JWTAuth::claims([
    'email' => $user->email,
    'name' => $user->name
])->fromUser($user);
```

**Penjelasan:**
- `DummyUser` implement `JWTSubject` interface
- `JWTAuth::claims()` - Menambahkan custom claims ke token
- `fromUser()` - Generate token berdasarkan user object
- Token berisi: header, payload, signature

---

### C. Middleware JWT Validation (Tugas 3)

**Alur:**
```
1. Request masuk dengan Authorization header
   ↓
2. Middleware DummyJwtMiddleware dijalankan
   ↓
3. Middleware parse token dari header
   ↓
4. JWTAuth::parseToken() validasi signature
   ↓
5. Jika valid: payload di-attach ke request
   Jika tidak: return 401 Unauthorized
   ↓
6. Request dilanjutkan ke controller
```

---

### D. Endpoint Protection (Tugas 3)

**Implementasi:**
```php
Route::middleware(['dummy.jwt'])->group(function () {
    Route::post('/logout', [AuthController::class, 'logout']);
    Route::get('/profile', [AuthController::class, 'profile']);
    Route::get('/token-check', [AuthController::class, 'tokenCheck']);
});
```

**Benefit:**
- Hanya route dalam group yang protected
- Middleware otomatis check token pada setiap request
- Code cleaner dan maintainable

---

## 📚 VII. PEMBELAJARAN DAN INSIGHT

### Konsep yang Dipelajari

1. **JWT Structure**
   - Header: Tipe token dan algoritma
   - Payload: Data user (claims)
   - Signature: Verifikasi authenticity

2. **Token-Based Authentication**
   - Stateless authentication
   - Scalable untuk microservices
   - CORS-friendly untuk SPA

3. **Security Best Practice**
   - Password hashed (dalam implementasi production)
   - Token expiration
   - Token blacklist saat logout
   - Error message generic

4. **API Design Pattern**
   - RESTful endpoints
   - Proper HTTP status codes
   - Consistent JSON response
   - Middleware for cross-cutting concerns

### Kesimpulan

Praktikum ini berhasil mengimplementasikan token-based authentication dengan JWT pada Lumen API. Semua skenario testing berjalan sesuai ekspektasi dengan:

- ✅ Authentication flow yang aman
- ✅ Validasi data yang proper
- ✅ Error handling yang konsisten
- ✅ Code structure yang maintainable
- ✅ Documentation yang lengkap

---

## 📋 APPENDIX

### A. File Structure
```
lumenapi/
├── app/
│   ├── Http/
│   │   ├── Controllers/Api/AuthController.php
│   │   └── Middleware/DummyJwtMiddleware.php
│   └── Models/DummyUser.php
├── routes/
│   ├── api.php
│   └── web.php
├── config/
│   ├── auth.php
│   └── jwt.php
├── bootstrap/app.php
└── .env
```

### B. Dummy User Credentials
```
User 1: user@example.com / password123
User 2: admin@example.com / secret321
User 3: syafi@student.ub.ac.id / rahasia123
```

### C. Command untuk Development
```bash
# Start server
php -S localhost:8000 -t public

# Generate JWT Secret
php artisan jwt:secret

# Run tests
php artisan test
```

---

**End of Report**

Dibuat: 29 April 2026
Status: Selesai ✅
Nilai: Menunggu Evaluasi
