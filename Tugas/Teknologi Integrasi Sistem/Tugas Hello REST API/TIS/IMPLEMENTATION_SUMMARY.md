# IMPLEMENTASI LATIHAN 1 - SUMMARY

## Status: ✅ SELESAI

Semua perubahan telah diimplementasikan pada project Student REST API.

---

## PERUBAHAN YANG DILAKUKAN

### 1. ✅ Validasi Baru Ditambahkan

**File:** `app/Http/Controllers/StudentController.php`

#### a) Method `store()` - Create
```php
'nama' => 'required|string|min:3|max:50',        // ✓ BARU: min:3
'mataKuliah' => 'required|array|min:1',          // ✓ BARU: min:1
```

#### b) Method `update()` - Update
```php
'nama' => 'sometimes|required|string|min:3|max:50',    // ✓ BARU: min:3
'mataKuliah' => 'sometimes|required|array|min:1',      // ✓ BARU: min:1
```

### 2. ✅ Response Structure Sudah Konsisten

**Struktur Sukses:**
```json
{
    "message": "Success message",
    "data": { ... }
}
```

**Struktur Gagal:**
```json
{
    "message": "Validation failed",
    "errors": {
        "field_name": ["error message"]
    }
}
```

### 3. ✅ 2 Endpoint Baru Ditambahkan

**File:** `routes/api.php`

#### a) Compound Data Endpoint
```php
Route::get('/students/{nim}', [StudentController::class, 'show']);
```
- **Method:** `show($nim)` - Baru
- **Response:** Mahasiswa + Mata Kuliah dalam 1 paket
- **Status:** 200 OK (jika ditemukan), 404 (jika tidak ditemukan)

#### b) Nested Resource Endpoint
```php
Route::get('/students/{nim}/mata-kuliah', [StudentController::class, 'mataKuliahByStudent']);
```
- **Method:** `mataKuliahByStudent($nim)` - Baru
- **Response:** HANYA array mata kuliah
- **Status:** 200 OK (jika ditemukan), 404 (jika tidak ditemukan)

### 4. ✅ 2 Method Controller Baru

#### Method Show - Compound Data
```php
public function show($nim)
{
    $students = $this->loadStudents();
    
    foreach ($students as $student) {
        if ($student['nim'] === $nim) {
            return response()->json([
                "message" => "Student retrieved successfully",
                "data" => $student
            ], 200);
        }
    }
    
    return response()->json([
        "message" => "Student not found",
        "error" => "NIM {$nim} tidak ditemukan"
    ], 404);
}
```

#### Method Mata Kuliah By Student - Nested Resource
```php
public function mataKuliahByStudent($nim)
{
    $students = $this->loadStudents();
    
    foreach ($students as $student) {
        if ($student['nim'] === $nim) {
            return response()->json([
                "message" => "Courses retrieved successfully",
                "student_nim" => $nim,
                "data" => $student['mataKuliah']
            ], 200);
        }
    }
    
    return response()->json([
        "message" => "Student not found",
        "error" => "NIM {$nim} tidak ditemukan"
    ], 404);
}
```

---

## FILE YANG TELAH DIMODIFIKASI

| File | Perubahan |
|------|-----------|
| `app/Http/Controllers/StudentController.php` | Validasi (min:3, min:1) + 2 method baru |
| `routes/api.php` | 2 route baru untuk show & mataKuliahByStudent |

---

## FILE YANG TELAH DIBUAT

| File | Deskripsi |
|------|-----------|
| `TESTING_GUIDE.md` | Panduan lengkap testing 5 skenario di Postman |
| `Postman_Collection.json` | Postman collection siap import untuk testing |

---

## SKENARIO TESTING (SIAP DIUJI)

### ✅ Skenario 1: POST Create Valid
- **Expected Status:** 201 Created
- **Validasi:** Nama 14 karakter (√), MK 2 items (√)

### ✅ Skenario 2A: POST Create Invalid (Nama Pendek)
- **Expected Status:** 422 Unprocessable Entity
- **Error:** "nama field must be at least 3 characters"

### ✅ Skenario 2B: POST Create Invalid (MK Kosong)
- **Expected Status:** 422 Unprocessable Entity
- **Error:** "mataKuliah field must have at least 1 items"

### ✅ Skenario 2C: POST Create Invalid (Kode MK Format Salah)
- **Expected Status:** 422 Unprocessable Entity
- **Error:** "mataKuliah.0.kode format is invalid"

### ✅ Skenario 3: PATCH/PUT Update Valid
- **Expected Status:** 200 OK
- **Response:** Full student list with updated data

### ✅ Skenario 4: GET Compound Data
- **Endpoint:** `GET /api/students/{nim}`
- **Expected:** Mahasiswa + Mata Kuliah (1 paket)

### ✅ Skenario 5: GET Nested Resource
- **Endpoint:** `GET /api/students/{nim}/mata-kuliah`
- **Expected:** HANYA array mata kuliah

---

## PENJELASAN: COMPOUND DATA vs NESTED RESOURCE

### Compound Data (Endpoint: `/api/students/{nim}`)

**Definisi:** Mengembalikan resource utama BESERTA sub-resource dalam satu response.

**Karakteristik:**
- URL: Flat/Simple (`/api/students/{nim}`)
- Response: Object lengkap dengan relasi sudah included
- Use Case: Dashboard, profile page yang butuh data lengkap
- Keuntungan: Single request, tidak perlu multiple calls
- Kerugian: Response lebih besar, kurang flexible jika hanya butuh sebagian

**Response Format:**
```json
{
    "message": "Student retrieved successfully",
    "data": {
        "nim": "245150707111012",
        "nama": "Citra Dewi",
        "mataKuliah": [
            { "kode": "CIE61205", "nama": "PemWeb", "sks": 3 },
            { "kode": "COM60015", "nama": "MatDis", "sks": 2 }
        ]
    }
}
```

---

### Nested Resource (Endpoint: `/api/students/{nim}/mata-kuliah`)

**Definisi:** Mengembalikan HANYA sub-resource dalam hierarki resource path.

**Karakteristik:**
- URL: Hierarchical/Nested (`/api/students/{nim}/mata-kuliah`)
- Response: HANYA array of mata kuliah, tanpa data siswa
- Use Case: Ketika user hanya butuh specific sub-resource
- Keuntangi: RESTful design, focused data, response lebih kecil
- Kerugian: Multiple requests jika butuh data lengkap

**Response Format:**
```json
{
    "message": "Courses retrieved successfully",
    "student_nim": "245150707111012",
    "data": [
        { "kode": "CIE61205", "nama": "PemWeb", "sks": 3 },
        { "kode": "COM60015", "nama": "MatDis", "sks": 2 }
    ]
}
```

---

## RINGKASAN PERBANDINGAN (TABEL)

| Parameter | Compound Data | Nested Resource |
|-----------|---------------|-----------------|
| **REST Standard** | ⭐⭐⭐☆☆ | ⭐⭐⭐⭐⭐ |
| **Fleksibilitas** | Rendah (semua data) | Tinggi (focused) |
| **Request Count** | Satu, data lengkap | Satu, data spesifik |
| **Response Size** | Besar | Kecil |
| **Use Case** | Profil/Dashboard | Filter/Detail Sub-Resource |
| **Network Efficient** | ❌ (mungkin over-fetch) | ✅ (minimal data) |
| **API Maturity** | ⭐⭐⭐⭐☆ | ⭐⭐⭐⭐⭐ |

---

## CARA TESTING DENGAN POSTMAN

### Opsi 1: Import Collection (Rekomendasi)
1. Buka Postman
2. Klik "Import"
3. Pilih file `Postman_Collection.json`
4. Pastikan `base_url` environment variable = `http://localhost:8000`
5. Jalankan semua request satu per satu

### Opsi 2: Manual Testing
1. Ikuti panduan di `TESTING_GUIDE.md`
2. Buat request manual sesuai skenario
3. Capture screenshot

---

## VALIDASI RULES REFERENCE

```
STORE METHOD:
├── nim: required|digits:15
├── nama: required|string|min:3|max:50
└── mataKuliah: required|array|min:1
    ├── kode: required|regex:/^[A-Z]{3}[0-9]{5}$/
    ├── nama: required|string|max:50
    └── sks: required|numeric|min:1|max:6

UPDATE METHOD:
├── nama: sometimes|required|string|min:3|max:50
└── mataKuliah: sometimes|required|array|min:1
    ├── kode: sometimes|required|regex:/^[A-Z]{3}[0-9]{5}$/
    ├── nama: sometimes|required|string|max:50
    └── sks: sometimes|required|numeric|min:1|max:6
```

---

## TODO - UNTUK PENGUMPULAN TUGAS

- [ ] Screenshot Skenario 1 (POST Create Valid, 201)
- [ ] Screenshot Skenario 2 (POST Create Invalid, 422)
- [ ] Screenshot Skenario 3 (PATCH/PUT Update, 200)
- [ ] Screenshot Skenario 4 (GET Compound Data)
- [ ] Screenshot Skenario 5 (GET Nested Resource)
- [ ] Jawaban: Penjelasan perbedaan Compound Data & Nested Resource
- [ ] Lampiran: Screenshot hasil 5 skenario testing

---

## QUICK START

```bash
# 1. Pastikan Laravel server berjalan
php artisan serve

# 2. Server akan run di http://localhost:8000

# 3. Buka Postman dan import Postman_Collection.json

# 4. Jalankan 5 skenario testing

# 5. Screenshot hasil dan dokumentasikan
```

---

## NOTES

- ✅ Validasi `min:3` untuk nama: Minimal 3 karakter
- ✅ Validasi `min:1` untuk mataKuliah: Array minimal 1 item
- ✅ Response format konsisten: `message` + `data` (sukses) / `errors` (gagal)
- ✅ Compound Data: `/api/students/{nim}` → data lengkap
- ✅ Nested Resource: `/api/students/{nim}/mata-kuliah` → hanya array MK
- ✅ Error handling: Status 422 untuk validation error
- ✅ RESTful design principles diterapkan

---

Semua implementasi **SIAP DITEST** dan **SIAP DIKUMPULKAN**! 🎉
