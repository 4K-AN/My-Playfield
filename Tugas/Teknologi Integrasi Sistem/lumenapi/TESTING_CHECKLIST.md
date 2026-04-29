# 📋 Checklist Testing API JWT - Lumen

## Persiapan Sebelum Testing

- [ ] Server PHP sudah berjalan (`php -S localhost:8000 -t public`)
- [ ] Postman sudah diinstall dan siap digunakan
- [ ] Environment BASE_URL sudah di-set ke `http://localhost:8000`
- [ ] Collection Postman sudah di-import dari `Lumen_JWT_API.postman_collection.json`

---

## Section 1: Testing Authentication (Tanpa Token)

### Test 1.1: Register - Berhasil ✅

**Request:**
```
POST http://localhost:8000/api/register
Content-Type: application/json

{
    "name": "Mahasiswa Baru",
    "email": "maba@student.ub.ac.id",
    "password": "password123",
    "password_confirmation": "password123"
}
```

**Expected Response:**
- Status: **201 Created**
- Body:
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

**Checklist:**
- [ ] Status code adalah 201
- [ ] Message berisi "successfully"
- [ ] ID user ter-generate
- [ ] Email sesuai dengan input

---

### Test 1.2: Register - Email Duplikat (Gagal) ❌

**Request:**
```
POST http://localhost:8000/api/register
Content-Type: application/json

{
    "name": "Duplicate User",
    "email": "user@example.com",
    "password": "password123",
    "password_confirmation": "password123"
}
```

**Expected Response:**
- Status: **422 Unprocessable Entity**
- Body:
```json
{
    "message": "Pendaftaran gagal, email sudah terdaftar di sistem."
}
```

**Checklist:**
- [ ] Status code adalah 422
- [ ] Message berisi "gagal"
- [ ] Email `user@example.com` adalah email yang sudah ada di dummy users

---

### Test 1.3: Login - Berhasil ✅

**Request:**
```
POST http://localhost:8000/api/login
Content-Type: application/json

{
    "email": "syafi@student.ub.ac.id",
    "password": "rahasia123"
}
```

**Expected Response:**
- Status: **200 OK**
- Body:
```json
{
    "message": "Login successful (dummy)",
    "token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
}
```

**Checklist:**
- [ ] Status code adalah 200
- [ ] Message berisi "successful"
- [ ] Token ada dan format dimulai dengan "eyJ"
- [ ] **PENTING: COPY TOKEN INI UNTUK TEST SELANJUTNYA**

**💾 Simpan Token di Postman:**
1. Buka tab "Tests"
2. Tambahkan script:
```javascript
pm.environment.set("jwt_token", pm.response.json().token);
```
3. Jalankan request, token akan otomatis tersimpan

---

### Test 1.4: Login - Password Salah (Gagal) ❌

**Request:**
```
POST http://localhost:8000/api/login
Content-Type: application/json

{
    "email": "syafi@student.ub.ac.id",
    "password": "passwordngawur"
}
```

**Expected Response:**
- Status: **401 Unauthorized**
- Body:
```json
{
    "message": "Invalid email or password"
}
```

**Checklist:**
- [ ] Status code adalah 401
- [ ] Message berisi "Invalid"
- [ ] Tidak ada token dalam response

---

## Section 2: Testing Protected Endpoints (Dengan Token)

### Test 2.1: Profile - Berhasil ✅

**Request:**
```
GET http://localhost:8000/api/profile
Authorization: Bearer <PASTE_TOKEN_DARI_TEST_1.3>
Accept: application/json
```

**Expected Response:**
- Status: **200 OK**
- Body:
```json
{
    "user": {
        "email": "syafi@student.ub.ac.id",
        "name": "Akhmad Syafiul Anam"
    }
}
```

**Checklist:**
- [ ] Status code adalah 200
- [ ] Email sesuai dengan token
- [ ] Name sesuai dengan user yang login

---

### Test 2.2: Token Check - Berhasil ✅

**Request:**
```
GET http://localhost:8000/api/token-check
Authorization: Bearer <PASTE_TOKEN_DARI_TEST_1.3>
Accept: application/json
```

**Expected Response:**
- Status: **200 OK**
- Body:
```json
{
    "message": "Token valid",
    "user": {
        "email": "syafi@student.ub.ac.id",
        "name": "Akhmad Syafiul Anam"
    }
}
```

**Checklist:**
- [ ] Status code adalah 200
- [ ] Message berisi "Token valid"
- [ ] User data ter-display dengan benar

---

### Test 2.3: Logout - Berhasil ✅

**Request:**
```
POST http://localhost:8000/api/logout
Authorization: Bearer <PASTE_TOKEN_DARI_TEST_1.3>
Accept: application/json
```

**Expected Response:**
- Status: **200 OK**
- Body:
```json
{
    "message": "User logged out successfully"
}
```

**Checklist:**
- [ ] Status code adalah 200
- [ ] Message berisi "logged out successfully"

---

### Test 2.4: Profile - Token Sudah Di-Logout (Gagal) ❌

**Request:**
```
GET http://localhost:8000/api/profile
Authorization: Bearer <TOKEN_YANG_SUDAH_DI_LOGOUT>
Accept: application/json
```

**Expected Response:**
- Status: **401 Unauthorized**
- Body:
```json
{
    "message": "Token invalid or expired",
    "error": "The token has been blacklisted"
}
```

**Checklist:**
- [ ] Status code adalah 401
- [ ] Message berisi "invalid"
- [ ] Token tidak bisa digunakan setelah logout

---

### Test 2.5: Profile - Tanpa Token (Gagal) ❌

**Request:**
```
GET http://localhost:8000/api/profile
Accept: application/json
```

**Expected Response:**
- Status: **401 Unauthorized**
- Body:
```json
{
    "message": "Token invalid or expired",
    "error": "The token could not be parsed from the request"
}
```

**Checklist:**
- [ ] Status code adalah 401
- [ ] Message berisi "Token invalid"
- [ ] Endpoint meminta token

---

### Test 2.6: Token Check - Invalid Token (Gagal) ❌

**Request:**
```
GET http://localhost:8000/api/token-check
Authorization: Bearer invalid_token_abc123
Accept: application/json
```

**Expected Response:**
- Status: **401 Unauthorized**
- Body:
```json
{
    "message": "Token invalid or expired",
    "error": "Malformed token"
}
```

**Checklist:**
- [ ] Status code adalah 401
- [ ] Message berisi "invalid"

---

## Section 3: Health Check

### Test 3.1: Ping - Server Hidup ✅

**Request:**
```
GET http://localhost:8000/api/ping
Accept: application/json
```

**Expected Response:**
- Status: **200 OK**
- Body:
```json
{
    "message": "pong"
}
```

**Checklist:**
- [ ] Status code adalah 200
- [ ] Message adalah "pong"
- [ ] Server sudah siap menerima request

---

## 📊 Summary Testing

| No | Test Case | Expected | Actual | Status |
|----|-----------|----------|--------|--------|
| 1.1 | Register Berhasil | 201 | | ✅/❌ |
| 1.2 | Register Duplikat | 422 | | ✅/❌ |
| 1.3 | Login Berhasil | 200 + Token | | ✅/❌ |
| 1.4 | Login Gagal | 401 | | ✅/❌ |
| 2.1 | Profile Berhasil | 200 | | ✅/❌ |
| 2.2 | Token Check | 200 | | ✅/❌ |
| 2.3 | Logout | 200 | | ✅/❌ |
| 2.4 | Token Logout Dipakai | 401 | | ✅/❌ |
| 2.5 | Profile Tanpa Token | 401 | | ✅/❌ |
| 2.6 | Invalid Token | 401 | | ✅/❌ |
| 3.1 | Ping | 200 | | ✅/❌ |

---

## 🔐 Header Reference

### Untuk Endpoint Tanpa Proteksi (Register, Login)
```http
Content-Type: application/json
```

### Untuk Endpoint Berproteksi (Profile, Logout, Token-Check)
```http
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...
Accept: application/json
```

---

## 💡 Troubleshooting

### Masalah: 404 Not Found
**Solusi:**
- [ ] Pastikan URL benar: `http://localhost:8000/api/...`
- [ ] Pastikan server PHP sudah running
- [ ] Cek routes di `routes/api.php`

### Masalah: 500 Internal Server Error
**Solusi:**
- [ ] Cek terminal PHP untuk error message
- [ ] Verifikasi syntax di AuthController
- [ ] Pastikan JWT package sudah terinstall

### Masalah: CORS Error
**Solusi:**
- [ ] Error ini normal untuk development
- [ ] Abaikan di Postman (Postman tidak enforce CORS)
- [ ] Configure CORS di production jika diperlukan

### Masalah: Token tidak ter-generate
**Solusi:**
- [ ] Cek JWT_SECRET di `.env` file
- [ ] Pastikan DummyUser class sudah benar
- [ ] Verifikasi AuthController logic

---

## 📚 Referensi File Kode

| File | Deskripsi |
|------|-----------|
| `app/Http/Controllers/Api/AuthController.php` | Controller untuk authentication logic |
| `app/Http/Middleware/DummyJwtMiddleware.php` | Middleware untuk JWT validation |
| `routes/api.php` | Routes definition untuk API endpoints |
| `config/auth.php` | Authentication configuration |
| `config/jwt.php` | JWT configuration |
| `.env` | Environment variables (JWT_SECRET) |

---

## ✅ Final Checklist

Semua test sudah berhasil?

- [ ] Section 1 (4 tests) - Semua ✅
- [ ] Section 2 (6 tests) - Semua ✅
- [ ] Section 3 (1 test) - Semua ✅
- [ ] Dokumentasi dibaca dan dipahami
- [ ] Kode sudah di-commit ke Git
- [ ] Siap untuk deployment

---

**Testing selesai! 🎉**

Jika ada yang belum sesuai, cek kembali file-file kode dan pastikan semuanya sudah updated sesuai dokumentasi.
