# PANDUAN TESTING API GATEWAY DENGAN POSTMAN

## 📌 SETUP ENVIRONMENT VARIABLES

Sebelum testing, setup variables berikut di Postman:

### Create Environment: "API Gateway Test"

| Variable | Initial Value | Current Value |
|----------|---------------|---------------|
| BASE_URL | http://127.0.0.1:8000 | http://127.0.0.1:8000 |
| USER_EMAIL | user@example.com | user@example.com |
| USER_PASSWORD | password123 | password123 |
| ADMIN_EMAIL | admin@example.com | admin@example.com |
| ADMIN_PASSWORD | secret321 | secret321 |
| USER_TOKEN | | (akan diisi otomatis) |
| ADMIN_TOKEN | | (akan diisi otomatis) |
| TEST_NIM | 245150707111099 | 245150707111099 |

---

## 🧪 TESTING STEPS

### **STEP 0: Verifikasi Server Berjalan**

**Request:**
```
GET http://127.0.0.1:8000/api/ping
```

**Expected Response:**
```json
{
  "message": "pong"
}
```

**Status:** 200 OK

---

## 🔵 TEST CASE 1: User Login dan Akses GET /api/gateway/students

### Skenario:
User berhasil login, mendapatkan token JWT, kemudian mengakses daftar student melalui gateway.

### Step 1.1: Login User

**Request:**
```http
POST http://127.0.0.1:8000/api/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "password123"
}
```

**Pre-request Script:**
```javascript
// None
```

**Tests Script (untuk auto-capture token):**
```javascript
if (pm.response.code === 200) {
    var jsonData = pm.response.json();
    pm.environment.set("USER_TOKEN", jsonData.token);
    console.log("✅ USER_TOKEN saved: " + jsonData.token.substring(0, 20) + "...");
}
```

**Expected Response:**
```json
{
  "message": "Login successful (dummy)",
  "token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
}
```

**Status:** 200 OK

**Verification:**
- ✓ Response berisi "Login successful"
- ✓ Token berhasil disimpan di environment variable

---

### Step 1.2: User Akses GET /api/gateway/students

**Request:**
```http
GET http://127.0.0.1:8000/api/gateway/students
Authorization: Bearer {{USER_TOKEN}}
Accept: application/json
```

**Pre-request Script:**
```javascript
console.log("Testing GET /api/gateway/students with USER token");
console.log("Token: " + pm.environment.get("USER_TOKEN").substring(0, 20) + "...");
```

**Tests Script:**
```javascript
pm.test("Status should be 200", function () {
    pm.response.to.have.status(200);
});

pm.test("Response should contain gateway info", function () {
    var jsonData = pm.response.json();
    pm.expect(jsonData.gateway).to.eql("API Gateway");
    pm.expect(jsonData.message).to.eql("Request forwarded to Student Service");
});

pm.test("Response should contain student data", function () {
    var jsonData = pm.response.json();
    pm.expect(jsonData.result).to.be.an('array');
});
```

**Expected Response:**
```json
{
  "gateway": "API Gateway",
  "message": "Request forwarded to Student Service",
  "result": [
    {
      "nim": "245150707111012",
      "nama": "Citra Dewi",
      "mataKuliah": [
        {
          "kode": "CIE61205",
          "nama": "PemWeb",
          "sks": 3
        },
        {
          "kode": "COM60015",
          "nama": "MatDis",
          "sks": 2
        }
      ]
    },
    {
      "nim": "245150707111013",
      "nama": "Andy Lau",
      "mataKuliah": [
        {
          "kode": "CIE61205",
          "nama": "PemWeb",
          "sks": 3
        },
        {
          "kode": "CIE61206",
          "nama": "JarKom",
          "sks": 3
        },
        {
          "kode": "CIE61208",
          "nama": "BasDat",
          "sks": 3
        }
      ]
    }
  ]
}
```

**Status:** 200 OK

**Verification:**
- ✓ Response status 200
- ✓ Gateway info tercantum
- ✓ Request berhasil diteruskan ke StudentController
- ✓ Data student tampil lengkap

---

## 🔴 TEST CASE 2: User Gagal Akses POST /api/gateway/students (Role Denied)

### Skenario:
User dengan role "user" mencoba membuat student baru melalui gateway. Gateway menolak karena endpoint POST hanya untuk admin.

### Step 2.1: User Mencoba POST Student

**Request:**
```http
POST http://127.0.0.1:8000/api/gateway/students
Authorization: Bearer {{USER_TOKEN}}
Content-Type: application/json

{
  "nim": "245150707111099",
  "nama": "Budi Santoso",
  "mataKuliah": [
    {
      "kode": "CIE61205",
      "nama": "PemWeb",
      "sks": 3
    }
  ]
}
```

**Pre-request Script:**
```javascript
console.log("Testing POST /api/gateway/students with USER token (should be denied)");
var token = pm.environment.get("USER_TOKEN");
console.log("User Token Role: user (only admin allowed)");
```

**Tests Script:**
```javascript
pm.test("Status should be 403 Forbidden", function () {
    pm.response.to.have.status(403);
});

pm.test("Response should contain access denied message", function () {
    var jsonData = pm.response.json();
    pm.expect(jsonData.message).to.include("Access denied");
    pm.expect(jsonData.message).to.include("role");
});
```

**Expected Response:**
```json
{
  "message": "Access denied. You do not have the required role."
}
```

**Status:** 403 Forbidden

**Verification:**
- ✓ Status 403 Forbidden
- ✓ Message menyebutkan access denied
- ✓ Role middleware berhasil mencegah user akses admin endpoint

---

## 🟢 TEST CASE 3: Admin Berhasil Akses POST /api/gateway/students

### Skenario:
Admin berhasil login, kemudian membuat student baru melalui gateway. Request berhasil diteruskan dan data disimpan.

### Step 3.1: Login Admin

**Request:**
```http
POST http://127.0.0.1:8000/api/login
Content-Type: application/json

{
  "email": "admin@example.com",
  "password": "secret321"
}
```

**Tests Script:**
```javascript
if (pm.response.code === 200) {
    var jsonData = pm.response.json();
    pm.environment.set("ADMIN_TOKEN", jsonData.token);
    console.log("✅ ADMIN_TOKEN saved: " + jsonData.token.substring(0, 20) + "...");
}
```

**Expected Response:**
```json
{
  "message": "Login successful (dummy)",
  "token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
}
```

**Status:** 200 OK

---

### Step 3.2: Admin POST Student via Gateway

**Request:**
```http
POST http://127.0.0.1:8000/api/gateway/students
Authorization: Bearer {{ADMIN_TOKEN}}
Content-Type: application/json

{
  "nim": "245150707111099",
  "nama": "Budi Santoso",
  "mataKuliah": [
    {
      "kode": "CIE61205",
      "nama": "PemWeb",
      "sks": 3
    }
  ]
}
```

**Pre-request Script:**
```javascript
console.log("Testing POST /api/gateway/students with ADMIN token (should succeed)");
console.log("Admin Token Role: admin (allowed for POST)");
```

**Tests Script:**
```javascript
pm.test("Status should be 201 Created", function () {
    pm.response.to.have.status(201);
});

pm.test("Response should indicate request forwarded", function () {
    var jsonData = pm.response.json();
    pm.expect(jsonData.gateway).to.eql("API Gateway");
    pm.expect(jsonData.message).to.eql("Request forwarded to Student Service");
});

pm.test("Response should contain created student data", function () {
    var jsonData = pm.response.json();
    var result = jsonData.result;
    pm.expect(result.message).to.include("created");
    pm.expect(result.data.nim).to.eql("245150707111099");
});
```

**Expected Response:**
```json
{
  "gateway": "API Gateway",
  "message": "Request forwarded to Student Service",
  "result": {
    "message": "Student created/updated successfully",
    "data": {
      "nim": "245150707111099",
      "nama": "Budi Santoso",
      "mataKuliah": [
        {
          "kode": "CIE61205",
          "nama": "PemWeb",
          "sks": 3
        }
      ]
    }
  }
}
```

**Status:** 201 Created

**Verification:**
- ✓ Status 201 Created
- ✓ Gateway berhasil forward request
- ✓ Student data berhasil disimpan
- ✓ Role authorization berhasil (admin dapat akses)

---

## 🚫 TEST CASE 4: Request Tanpa Token Ditolak (401 Unauthorized)

### Skenario:
Client mengirim request ke gateway TANPA mengirim Authorization header. Gateway menolak dengan status 401.

### Step 4.1: GET /api/gateway/students WITHOUT Token

**Request:**
```http
GET http://127.0.0.1:8000/api/gateway/students
Accept: application/json
```

**⚠️ PENTING:** Jangan include Authorization header!

**Tests Script:**
```javascript
pm.test("Status should be 401 Unauthorized", function () {
    pm.response.to.have.status(401);
});

pm.test("Response should contain token required message", function () {
    var jsonData = pm.response.json();
    pm.expect(jsonData.message).to.include("Token");
});
```

**Expected Response:**
```json
{
  "message": "Token is required"
}
```

**Status:** 401 Unauthorized

**Verification:**
- ✓ Status 401 Unauthorized
- ✓ DummyJwtMiddleware berhasil mendeteksi missing token
- ✓ Request ditolak sebelum sampai ke GatewayController

---

## ⚠️ TEST CASE 5: Token Valid tapi Role Tidak Sesuai (403 Forbidden)

### Skenario:
Token valid (user sudah login), tapi role user tidak cocok dengan endpoint yang memerlukan admin. Role middleware menolak dengan status 403.

### Step 5.1: User Mencoba DELETE Student (Admin Only)

**Request:**
```http
DELETE http://127.0.0.1:8000/api/gateway/students/245150707111012
Authorization: Bearer {{USER_TOKEN}}
Accept: application/json
```

**⚠️ PENTING:** Gunakan USER_TOKEN (role: user), bukan ADMIN_TOKEN!

**Pre-request Script:**
```javascript
console.log("Testing DELETE /api/gateway/students/{nim} with USER token (should be denied)");
console.log("Endpoint DELETE hanya untuk role: admin");
console.log("User role: user");
```

**Tests Script:**
```javascript
pm.test("Status should be 403 Forbidden", function () {
    pm.response.to.have.status(403);
});

pm.test("Response should contain access denied message", function () {
    var jsonData = pm.response.json();
    pm.expect(jsonData.message).to.include("Access denied");
    pm.expect(jsonData.message).to.include("role");
});

pm.test("Token should be valid (not expired)", function () {
    // If we got 403 instead of 401, it means token is valid but role denied
    pm.response.to.not.have.status(401);
});
```

**Expected Response:**
```json
{
  "message": "Access denied. You do not have the required role."
}
```

**Status:** 403 Forbidden

**Verification:**
- ✓ Status 403 Forbidden (bukan 401, berarti token valid)
- ✓ Token berhasil divalidasi
- ✓ Role user ditolak untuk endpoint DELETE
- ✓ RoleMiddleware berhasil mencegah akses yang tidak sesuai role

---

## 📊 RINGKASAN HASIL TESTING

| Test Case | Method | Endpoint | Role | Expected Status | Result |
|-----------|--------|----------|------|-----------------|--------|
| 1 | GET | /api/gateway/students | user | 200 | ✅ PASS |
| 2 | POST | /api/gateway/students | user | 403 | ✅ PASS |
| 3 | POST | /api/gateway/students | admin | 201 | ✅ PASS |
| 4 | GET | /api/gateway/students | (none) | 401 | ✅ PASS |
| 5 | DELETE | /api/gateway/students/{nim} | user | 403 | ✅ PASS |

---

## 🔧 POSTMAN COLLECTION EXPORT

Jika ingin mengekspor collection untuk sharing, gunakan:
1. Click "..." di collection name
2. Select "Export"
3. Choose "Collection v2.1"
4. Save sebagai JSON file

---

## 📝 TROUBLESHOOTING TESTING

### Problem: "Invalid token" pada semua gateway requests
**Solution:**
- Pastikan sudah login terlebih dahulu
- Verifikasi token disimpan di USER_TOKEN atau ADMIN_TOKEN variable
- Check format Authorization header: `Bearer {token}` (ada spasi)

### Problem: Test Case 2 mengembalikan 500 instead of 403
**Solution:**
- Verify RoleMiddleware registered di `app/Http/Kernel.php`
- Check RoleMiddleware syntax
- Clear cache: `php artisan cache:clear`

### Problem: "Student created/updated successfully" muncul di test case 1
**Solution:**
- Expected, karena /api/students endpoint (direct) tidak ada middleware
- Gateway endpoint sudah bekerja, wrapping data dengan gateway info

### Problem: Cannot see log entries
**Solution:**
```bash
# Clear log file
echo "" > storage/logs/laravel.log

# Monitor real-time
tail -f storage/logs/laravel.log

# View last entries
tail -50 storage/logs/laravel.log
```

---

## ✅ COMPLETE TESTING FLOW

Jalankan requests dalam urutan ini untuk hasil optimal:

1. **STEP 0:** Verify ping (/api/ping) → 200 pong
2. **TEST CASE 1:**
   - 1.1: Login User → Save USER_TOKEN
   - 1.2: GET /api/gateway/students with USER_TOKEN → 200 Success
3. **TEST CASE 2:**
   - 2.1: POST /api/gateway/students with USER_TOKEN → 403 Denied
4. **TEST CASE 3:**
   - 3.1: Login Admin → Save ADMIN_TOKEN
   - 3.2: POST /api/gateway/students with ADMIN_TOKEN → 201 Created
5. **TEST CASE 4:**
   - 4.1: GET /api/gateway/students WITHOUT Token → 401 Unauthorized
6. **TEST CASE 5:**
   - 5.1: DELETE /api/gateway/students/{nim} with USER_TOKEN → 403 Denied

**Total Requests:** 8 requests
**Expected Results:** All 5 test cases PASS ✅

---

## 📚 ADDITIONAL ENDPOINTS TO TEST

Setelah 5 test cases selesai, coba endpoints tambahan:

### Test Profile Gateway
```http
GET http://127.0.0.1:8000/api/gateway/profile
Authorization: Bearer {{USER_TOKEN}}
```

Expected: 200 dengan user info

### Test Admin Dashboard Gateway
```http
GET http://127.0.0.1:8000/api/gateway/admin/dashboard
Authorization: Bearer {{ADMIN_TOKEN}}
```

Expected: 200 dengan welcome message

### Test User Dashboard Gateway
```http
GET http://127.0.0.1:8000/api/gateway/user/dashboard
Authorization: Bearer {{USER_TOKEN}}
```

Expected: 200 dengan welcome message

---

**Happy Testing! 🚀**
