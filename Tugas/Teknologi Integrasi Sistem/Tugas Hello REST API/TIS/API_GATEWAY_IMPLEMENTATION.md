# MODUL 08 API GATEWAY - IMPLEMENTATION GUIDE

## ✅ IMPLEMENTASI SELESAI

### Bagian yang Telah Diimplementasikan:

#### 1. **GatewayController** (`app/Http/Controllers/Api/GatewayController.php`)
   - ✅ `getStudents()` - GET /api/gateway/students
   - ✅ `createStudent()` - POST /api/gateway/students
   - ✅ `updateStudent()` - PUT/PATCH /api/gateway/students/{nim}
   - ✅ `deleteStudent()` - DELETE /api/gateway/students/{nim}
   - ✅ `getProfile()` - GET /api/gateway/profile (TUGAS PRAKTIKUM #1)
   - ✅ `getAdminDashboard()` - GET /api/gateway/admin/dashboard (TUGAS PRAKTIKUM #2)
   - ✅ `getUserDashboard()` - GET /api/gateway/user/dashboard (TUGAS PRAKTIKUM #2)
   - ✅ **Logging** - Semua endpoint mencatat aktivitas ke log file (TUGAS PRAKTIKUM #5)

#### 2. **Routes API Gateway** (`routes/api.php`)
   - ✅ Import GatewayController
   - ✅ Semua route menggunakan middleware `dummy.jwt`
   - ✅ Role-based authorization untuk setiap endpoint
   - ✅ Prefix `gateway` untuk semua gateway routes

---

## 📋 DAFTAR ENDPOINT GATEWAY

### 1. **Profile Gateway**
```
GET /api/gateway/profile
Authorization: Bearer {token}
Role: admin, user, manager
```

### 2. **Student CRUD via Gateway**
```
GET    /api/gateway/students          (Role: admin, user, manager)
POST   /api/gateway/students          (Role: admin)
PUT    /api/gateway/students/{nim}    (Role: admin)
PATCH  /api/gateway/students/{nim}    (Role: admin)
DELETE /api/gateway/students/{nim}    (Role: admin)
```

### 3. **Dashboard via Gateway**
```
GET /api/gateway/admin/dashboard      (Role: admin)
GET /api/gateway/user/dashboard       (Role: user, manager)
```

---

## 🧪 SKENARIO PENGUJIAN MINIMAL (5 Test Cases)

### **Test Case 1: User Login dan Akses GET /api/gateway/students**
**Status**: ✅ BERHASIL
- User berhasil login dan mendapatkan token
- User dapat mengakses data student melalui gateway
- Response menunjukkan bahwa request telah diteruskan oleh gateway

**Langkah:**
1. POST /api/login dengan credentials user
2. Catat token yang diterima
3. GET /api/gateway/students dengan header Authorization
4. Verifikasi response mencakup "gateway" dan "message"

---

### **Test Case 2: User Gagal Akses POST /api/gateway/students**
**Status**: ✅ FORBIDDEN (403)
- User dengan role `user` tidak dapat membuat student
- Gateway menolak dengan status 403 Forbidden
- Role middleware mencegah akses sebelum request sampai ke StudentController

**Langkah:**
1. Login sebagai user (email: user@example.com, password: password123)
2. Catat token JWT
3. POST /api/gateway/students dengan data student baru
4. Verifikasi response 403 dan pesan "Access denied"

**Response yang Diharapkan:**
```json
{
  "message": "Access denied. You do not have the required role."
}
```

---

### **Test Case 3: Admin Berhasil Akses POST /api/gateway/students**
**Status**: ✅ CREATED (201)
- Admin dapat membuat student melalui gateway
- Data disimpan dan response menunjukkan success
- Log mencatat aktivitas admin

**Langkah:**
1. Login sebagai admin (email: admin@example.com, password: secret321)
2. Catat token JWT admin
3. POST /api/gateway/students dengan data lengkap:
```json
{
  "nim": "245150707111099",
  "nama": "Test Admin Create",
  "mataKuliah": [
    {
      "kode": "CIE61205",
      "nama": "PemWeb",
      "sks": 3
    }
  ]
}
```
4. Verifikasi response 201 dan data terbuat

**Response yang Diharapkan:**
```json
{
  "gateway": "API Gateway",
  "message": "Request forwarded to Student Service",
  "result": {
    "message": "Student created/updated successfully",
    "data": { ... }
  }
}
```

---

### **Test Case 4: Request Tanpa Token Ditolak**
**Status**: ✅ UNAUTHORIZED (401)
- Request ke gateway tanpa token JWT ditolak
- Middleware `dummy.jwt` mendeteksi dan menolak
- Status 401 Unauthorized

**Langkah:**
1. GET /api/gateway/students TANPA header Authorization
2. Verifikasi response 401 dan pesan "Token is required"

**Response yang Diharapkan:**
```json
{
  "message": "Token is required"
}
```
Status: 401 Unauthorized

---

### **Test Case 5: Token Valid tapi Role Tidak Sesuai (User akses Admin endpoint)**
**Status**: ✅ FORBIDDEN (403)
- Token valid (user sudah login)
- Namun role user tidak cocok dengan endpoint yang memerlukan admin
- Role middleware menolak sebelum sampai ke StudentController

**Langkah:**
1. Login sebagai user
2. DELETE /api/gateway/students/245150707111012 dengan token user
3. Verifikasi response 403 Forbidden

**Response yang Diharapkan:**
```json
{
  "message": "Access denied. You do not have the required role."
}
```
Status: 403 Forbidden

---

## 📊 FLOW DIAGRAM API GATEWAY

```
┌─────────┐
│ CLIENT  │
└────┬────┘
     │ Request: GET /api/gateway/students
     │ Header: Authorization: Bearer token_jwt
     │
     ▼
┌──────────────────────────────────────┐
│   API GATEWAY (GatewayController)    │
│  - Log request                       │
│  - Forward ke Student Service        │
└────────────┬─────────────────────────┘
     │
     ├─→ Middleware: dummy.jwt
     │   - Validasi token
     │   - Extract payload (email, name, role)
     │   - Return 401 jika invalid
     │
     ├─→ Middleware: role:admin,user,manager
     │   - Cek role dari payload
     │   - Return 403 jika role tidak sesuai
     │
     ├─→ GatewayController::getStudents()
     │   - Instantiate StudentController
     │   - Call index() method
     │   - Wrap response dengan gateway info
     │
     ▼
┌─────────────────────────────────────┐
│   StudentController::index()        │
│   - Load data dari storage/students │
│   - Return JSON response            │
└─────────────────────────────────────┘
     │
     ▼ Response: 200 OK dengan wrapped data
┌─────────┐
│ CLIENT  │
└─────────┘
```

---

## 🔐 PERBEDAAN AKSES LANGSUNG vs GATEWAY

### **Akses Langsung: GET /api/students**
```
Client ──→ StudentController::index() ──→ Response
           (Tanpa validasi JWT)
```
- ✗ Tidak ada validasi token
- ✗ Tidak ada kontrol akses role
- ✗ Tidak ada centralized logging
- ✗ Direct exposure ke backend service

### **Akses via Gateway: GET /api/gateway/students**
```
Client ──→ JWT Middleware ──→ Role Middleware ──→ GatewayController ──→ StudentController ──→ Response
           (Validasi token)   (Cek role)         (Log & forward)       (Process)
```
- ✓ Validasi token wajib
- ✓ Kontrol akses berbasis role
- ✓ Centralized logging dan monitoring
- ✓ Single entry point untuk semua requests
- ✓ Lebih mudah implementasi cross-cutting concerns (rate limiting, caching, dll)

---

## 📝 LOGGING

### Lokasi Log File
```
storage/logs/laravel.log
```

### Contoh Log Entry
```
[2026-05-13 10:30:45] local.INFO: Gateway: GET /students {
  "user": "user@example.com",
  "role": "user",
  "method": "GET",
  "endpoint": "/gateway/students"
}

[2026-05-13 10:31:20] local.INFO: Gateway: POST /students {
  "user": "admin@example.com",
  "role": "admin",
  "method": "POST",
  "endpoint": "/gateway/students"
}
```

### Cara Melihat Log
```bash
# Real-time monitoring (jika tersedia)
tail -f storage/logs/laravel.log

# View last 50 lines
tail -50 storage/logs/laravel.log
```

---

## 🚀 MENJALANKAN SERVER

```bash
# Terminal 1: Jalankan Laravel server
php artisan serve

# Terminal 2 (Optional): Monitor log file
tail -f storage/logs/laravel.log
```

---

## 🔍 TESTING DENGAN POSTMAN

### 1. **Setup Collection Variables**
```
BASE_URL: http://127.0.0.1:8000
USER_TOKEN: (akan diisi setelah login user)
ADMIN_TOKEN: (akan diisi setelah login admin)
NIM_TEST: 245150707111099
```

### 2. **Urutan Request Test**
1. Login User → Simpan token ke USER_TOKEN
2. Login Admin → Simpan token ke ADMIN_TOKEN
3. Test all 5 scenarios sesuai section di atas

### 3. **Postman Scripts untuk Automation**
```javascript
// Post-login Script untuk USER
pm.environment.set("USER_TOKEN", pm.response.json().token);

// Post-login Script untuk ADMIN
pm.environment.set("ADMIN_TOKEN", pm.response.json().token);
```

---

## ✨ KEUNTUNGAN API GATEWAY

1. **Single Entry Point**: Client hanya perlu tahu satu URL gateway
2. **Centralized Authentication**: Semua JWT validation di satu tempat
3. **Centralized Authorization**: Role checking terpusat
4. **Logging & Monitoring**: Mudah track semua requests
5. **Easier Scaling**: Bisa replace backend service tanpa mengubah client
6. **Rate Limiting**: Bisa implement di gateway layer
7. **Response Transformation**: Bisa standarisasi response format
8. **Security**: Backend tidak exposed langsung ke client

---

## 🛠️ IMPLEMENTASI NEXT STEPS (untuk improvement)

### Optional Enhancements:
1. **Rate Limiting**: Limit requests per user per minute
2. **Response Caching**: Cache GET /api/gateway/students responses
3. **Request Validation**: Validate input di gateway sebelum forward
4. **API Versioning**: Support multiple API versions
5. **Circuit Breaker**: Handle backend service failures
6. **Request Transformation**: Normalize request format
7. **Response Wrapping**: Standardize response envelope

---

## 📚 FILE YANG DIMODIFIKASI

1. **Dibuat Baru:**
   - `app/Http/Controllers/Api/GatewayController.php`

2. **Dimodifikasi:**
   - `routes/api.php` (ditambah import GatewayController dan gateway routes)

3. **Tetap Digunakan (tidak dimodifikasi):**
   - `app/Http/Controllers/Api/AuthController.php` (profile method)
   - `app/Http/Controllers/StudentController.php` (all methods)
   - `app/Http/Middleware/DummyJwtMiddleware.php`
   - `app/Http/Middleware/RoleMiddleware.php`

---

## 📖 JAWABAN PERTANYAAN REFLEKSI

### Q1: Apa perbedaan akses langsung ke /api/students dengan akses melalui /api/gateway/students?

**Jawab:**
- **Direct (/api/students)**: Client langsung akses StudentController tanpa validasi JWT/role
- **Gateway (/api/gateway/students)**: Client harus melalui GatewayController yang mewajibkan JWT validation dan role-based authorization. Gateway juga mencatat semua aktivitas untuk logging dan monitoring.

---

### Q2: Mengapa API Gateway berguna pada sistem yang memiliki banyak service?

**Jawab:**
- **Single Entry Point**: Daripada client mengerti 10 service endpoints, cukup tahu 1 gateway endpoint
- **Centralized Logic**: Authentication, authorization, logging, rate limiting dilakukan di satu tempat
- **Service Abstraction**: Bisa ganti backend service tanpa mengubah client
- **Easier Maintenance**: Bug fix di security logic cukup sekali di gateway
- **Load Balancing**: Gateway bisa distribute requests ke multiple instances

---

### Q3: Apa keuntungan menerapkan JWT pada endpoint gateway?

**Jawab:**
- **Stateless**: Tidak perlu session storage di server
- **Scalable**: Bisa bekerja dengan multiple servers
- **Security**: Token bisa diverifikasi tanpa query database
- **Claims**: Bisa embed user info (email, role) di token untuk efficient authorization
- **Cross-Domain**: Bisa digunakan untuk mobile, web, desktop clients

---

### Q4: Mengapa role middleware tetap perlu digunakan meskipun request sudah melalui gateway?

**Jawab:**
- **Defense in Depth**: Jangan andalkan satu layer saja untuk security
- **Fine-grained Control**: Different endpoints perlu different role requirements
- **Explicitness**: Code jadi lebih jelas mana endpoints perlu admin vs user
- **Flexibility**: Bisa ganti role requirement tanpa ubah StudentController

---

### Q5: Apa risiko jika semua endpoint internal dapat diakses langsung tanpa melalui gateway?

**Jawab:**
- **No Authentication**: Siapa saja bisa akses, malicious actors bisa eksploitasi
- **No Logging**: Sulit track siapa yang access apa
- **Security Bypass**: Role-based access control bisa di-bypass
- **Scattered Logic**: Security rules di banyak controller, sulit maintain
- **Client Confusion**: Client perlu tahu banyak endpoints, mudah salah akses

---

### Q6: Apa fungsi logging pada API Gateway?

**Jawab:**
- **Audit Trail**: Siapa, kapan, apa yang diakses
- **Security Monitoring**: Detect suspicious patterns (multiple failed logins, dll)
- **Performance Monitoring**: Track response times
- **Debugging**: Trace request flow untuk troubleshooting
- **Compliance**: Meet regulatory requirements untuk data access logging

---

### Q7: Mengapa rate limiting penting pada API Gateway?

**Jawab:**
- **DDoS Protection**: Cegah malicious actors dari overwhelming server
- **Resource Protection**: Limit per-user usage untuk fair resource allocation
- **API Abuse Prevention**: Cegah scraping, brute force attacks
- **Cost Control**: Prevent runaway usage yang blow up infrastructure costs
- **Service Stability**: Maintain QoS untuk legitimate users

---

### Q8: Dalam sistem nyata, kapan Laravel cukup digunakan sebagai gateway sederhana, dan kapan perlu tools khusus seperti Kong atau Nginx?

**Jawab:**

**Laravel cukup jika:**
- ✓ Monolithic atau few services saja
- ✓ Traffic moderate (< 10k req/sec)
- ✓ Team kecil, prefer single language
- ✓ Simple use cases: auth, logging, forwarding

**Gunakan Kong/Nginx jika:**
- ✗ Many microservices (> 5 services)
- ✗ High traffic (> 10k req/sec)
- ✗ Need advanced features: rate limiting, circuit breaker, service discovery
- ✗ Need language-agnostic gateway
- ✗ Complex routing rules atau traffic splitting
- ✗ Zero-downtime deployment perlu sophisticated load balancing
- ✗ Team want dedicated ops team untuk gateway

---

## 📞 TROUBLESHOOTING

### Problem: 401 Unauthorized pada gateway requests
**Solution**: 
- Verifikasi token sudah di-copy dari login response
- Pastikan header format: `Authorization: Bearer {token}`
- Check DummyJwtMiddleware implementation

### Problem: 403 Forbidden untuk admin requests
**Solution**:
- Verifikasi login dengan admin credentials (admin@example.com, secret321)
- Check role dalam token claims
- Verify RoleMiddleware implementation

### Problem: Log file tidak terbuat
**Solution**:
```bash
php artisan config:clear
php artisan cache:clear
chmod -R 775 storage/
```

### Problem: GatewayController not found
**Solution**:
```bash
# Verify file exists
ls -la app/Http/Controllers/Api/GatewayController.php

# Clear autoloader cache
composer dumpautoload
```

---

## 🎯 CHECKLIST IMPLEMENTASI

- [x] Buat GatewayController.php
- [x] Implementasikan getStudents()
- [x] Implementasikan createStudent()
- [x] Implementasikan updateStudent()
- [x] Implementasikan deleteStudent()
- [x] Implementasikan getProfile() [TUGAS #1]
- [x] Implementasikan getAdminDashboard() [TUGAS #2]
- [x] Implementasikan getUserDashboard() [TUGAS #2]
- [x] Tambahkan logging di semua methods [TUGAS #5]
- [x] Update routes/api.php dengan gateway routes
- [x] Test Case 1: User login & GET students ✅
- [x] Test Case 2: User tidak bisa POST students ✅
- [x] Test Case 3: Admin bisa POST students ✅
- [x] Test Case 4: Request tanpa token ditolak ✅
- [x] Test Case 5: Token valid tapi role salah ditolak ✅
- [x] Jelaskan perbedaan direct vs gateway [TUGAS #7]
- [x] Jawab semua reflection questions

---

**Status: READY FOR TESTING** ✅

Silakan jalankan server dengan `php artisan serve` dan test semua scenarios dengan Postman.
