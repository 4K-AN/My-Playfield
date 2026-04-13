# LATIHAN 1 - TUGAS PRAKTIKUM: REST API dengan Validasi & Nested Resources

## Ringkasan Perubahan

### 1. Validasi Baru yang Ditambahkan
- **Nama**: Minimal 3 karakter (`min:3`)
- **Mata Kuliah**: Minimal 1 mata kuliah (`min:1`) - array tidak boleh kosong

### 2. Struktur Response
**Sukses:**
```json
{
    "message": "Success message",
    "data": { ... }
}
```

**Gagal Validasi:**
```json
{
    "message": "Validation failed",
    "errors": { ... }
}
```

### 3. Endpoint Baru
- **GET /api/students/{nim}** - Compound Data (satu mahasiswa + mata kuliahnya)
- **GET /api/students/{nim}/mata-kuliah** - Nested Resource (hanya mata kuliah)

---

## 5 SKENARIO PENGUJIAN DI POSTMAN

### Skenario 1: POST Create Data Valid (Status 201 Created)
**URL:** `POST http://localhost:8000/api/students`

**Request Header:**
```
Content-Type: application/json
Accept: application/json
```

**Request Body:**
```json
{
    "nim": "245150707111014",
    "nama": "Muhammad Rizki",
    "mataKuliah": [
        {
            "kode": "CIE61205",
            "nama": "Pemrograman Web",
            "sks": 3
        },
        {
            "kode": "COM60015",
            "nama": "Matematika Diskrit",
            "sks": 2
        }
    ]
}
```

**Expected Response (Status: 201):**
```json
{
    "message": "Student created/updated successfully",
    "data": {
        "nim": "245150707111014",
        "nama": "Muhammad Rizki",
        "mataKuliah": [
            {
                "kode": "CIE61205",
                "nama": "Pemrograman Web",
                "sks": 3
            },
            {
                "kode": "COM60015",
                "nama": "Matematika Diskrit",
                "sks": 2
            }
        ]
    }
}
```

**Catatan:** ✓ Nama 14 karakter (lebih dari 3) ✓ Mata Kuliah 2 item (lebih dari 1)

---

### Skenario 2: POST Create Data Tidak Valid (Status 422 Unprocessable Entity)

#### 2A. Nama Terlalu Pendek (2 karakter)
**URL:** `POST http://localhost:8000/api/students`

**Request Body:**
```json
{
    "nim": "245150707111015",
    "nama": "AB",
    "mataKuliah": [
        {
            "kode": "CIE61205",
            "nama": "Pemrograman Web",
            "sks": 3
        }
    ]
}
```

**Expected Response (Status: 422):**
```json
{
    "message": "Validation failed",
    "errors": {
        "nama": [
            "The nama field must be at least 3 characters."
        ]
    }
}
```

#### 2B. Mata Kuliah Kosong (Array kosong)
**URL:** `POST http://localhost:8000/api/students`

**Request Body:**
```json
{
    "nim": "245150707111015",
    "nama": "Budi Santoso",
    "mataKuliah": []
}
```

**Expected Response (Status: 422):**
```json
{
    "message": "Validation failed",
    "errors": {
        "mataKuliah": [
            "The mataKuliah field must have at least 1 items."
        ]
    }
}
```

#### 2C. Kode Mata Kuliah Format Salah
**URL:** `POST http://localhost:8000/api/students`

**Request Body:**
```json
{
    "nim": "245150707111015",
    "nama": "Sri Rahayu",
    "mataKuliah": [
        {
            "kode": "CIE6120",
            "nama": "Pemrograman Web",
            "sks": 3
        }
    ]
}
```

**Expected Response (Status: 422):**
```json
{
    "message": "Validation failed",
    "errors": {
        "mataKuliah.0.kode": [
            "The mataKuliah.0.kode format is invalid."
        ]
    }
}
```

---

### Skenario 3: PATCH/PUT Update Data Valid (Status 200 OK)

**URL:** `PATCH http://localhost:8000/api/students/245150707111014`

**Request Body:**
```json
{
    "nama": "Muhammad Rizki Prabowo",
    "mataKuliah": [
        {
            "kode": "CIE61205",
            "nama": "Pemrograman Web",
            "sks": 3
        },
        {
            "kode": "COM60015",
            "nama": "Matematika Diskrit",
            "sks": 2
        },
        {
            "kode": "CIE61206",
            "nama": "Jaringan Komputer",
            "sks": 3
        }
    ]
}
```

**Expected Response (Status: 200):**
```json
{
    "message": "Student 245150707111014 updated successfully",
    "data": [
        {
            "nim": "245150707111012",
            "nama": "Citra Dewi",
            "mataKuliah": [...]
        },
        {
            "nim": "245150707111013",
            "nama": "Andy Lau",
            "mataKuliah": [...]
        },
        {
            "nim": "245150707111014",
            "nama": "Muhammad Rizki Prabowo",
            "mataKuliah": [
                {
                    "kode": "CIE61205",
                    "nama": "Pemrograman Web",
                    "sks": 3
                },
                {
                    "kode": "COM60015",
                    "nama": "Matematika Diskrit",
                    "sks": 2
                },
                {
                    "kode": "CIE61206",
                    "nama": "Jaringan Komputer",
                    "sks": 3
                }
            ]
        }
    ]
}
```

---

### Skenario 4: GET Ambil Satu Data Mahasiswa (Compound Data)

**URL:** `GET http://localhost:8000/api/students/245150707111012`

**Expected Response (Status: 200):**
```json
{
    "message": "Student retrieved successfully",
    "data": {
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
    }
}
```

**Penjelasan COMPOUND DATA:**
- Endpoint: `/api/students/{nim}`
- **Response menampilkan seluruh data mahasiswa + mata kuliahnya dalam 1 paket**
- Struktur: `data` berisi object dengan `nim`, `nama`, dan `mataKuliah[]`
- Keuntungan: Single request untuk mendapat info lengkap
- Kerugian: Tidak fleksibel jika hanya butuh data tertentu

---

### Skenario 5: GET Ambil Daftar Mata Kuliah Berdasarkan NIM (Nested Resource)

**URL:** `GET http://localhost:8000/api/students/245150707111012/mata-kuliah`

**Expected Response (Status: 200):**
```json
{
    "message": "Courses retrieved successfully",
    "student_nim": "245150707111012",
    "data": [
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
}
```

**Penjelasan NESTED RESOURCE:**
- Endpoint: `/api/students/{nim}/mata-kuliah` (URL bersarang/nested)
- **Response HANYA berisi array mata kuliah, tanpa data mahasiswa**
- Struktur: `data` adalah array of mata kuliah
- Keuntungan: Focused & RESTful, hanya data yang diperlukan
- Design pattern RESTful yang lebih clean

---

## PERBEDAAN COMPOUND DATA vs NESTED RESOURCE

| Aspek | Compound Data | Nested Resource |
|-------|---------------|-----------------|
| **Endpoint** | `/api/students/{nim}` | `/api/students/{nim}/mata-kuliah` |
| **Struktur URL** | Flat/Simple | Nested/Hierarchical |
| **Response Data** | Mahasiswa + Mata Kuliah dalam 1 object | Hanya array Mata Kuliah |
| **Request** | Single request mendapat semua | Single request hanya dependent resource |
| **Use Case** | Dashboard yang butuh data lengkap | Spesifik mendapat sub-resource |
| **REST Principle** | Less RESTful | More RESTful |
| **Flexibility** | Kurang fleksibel | Lebih fleksibel |
| **Network** | Satu paket besar | Data-focused, lebih kecil |

### Contoh Real-World:
**Compound Data - Profil User Lengkap:**
```
GET /api/users/123
Response: User name, email, address, phone, preferences, payments, orders, dll
```

**Nested Resource - Hanya Orders User:**
```
GET /api/users/123/orders
Response: Hanya array orders milik user 123
```

---

## LANGKAH-LANGKAH PENGUJIAN DI POSTMAN

### Persiapan:
1. Buka Postman
2. Buat collection baru: "Student REST API"
3. Tambahkan environment variable untuk base URL: `{{base_url}}` = `http://localhost:8000`

### Eksekusi Skenario:

1. **Buat request untuk Skenario 1** (Create Valid)
   - Method: `POST`
   - URL: `{{base_url}}/api/students`
   - Body: JSON dari Skenario 1
   - Klik "Send"
   - Capture screenshot (Status 201, response sesuai)

2. **Buat request untuk Skenario 2A** (Nama Pendek)
   - Method: `POST`
   - URL: `{{base_url}}/api/students`
   - Body: JSON Skenario 2A
   - Klik "Send"
   - Capture screenshot (Status 422, error message)

3. **Buat request untuk Skenario 3** (Update Valid)
   - Method: `PATCH`
   - URL: `{{base_url}}/api/students/245150707111014`
   - Body: JSON dari Skenario 3
   - Klik "Send"
   - Capture screenshot (Status 200)

4. **Buat request untuk Skenario 4** (Compound Data)
   - Method: `GET`
   - URL: `{{base_url}}/api/students/245150707111012`
   - Klik "Send"
   - Capture screenshot (menunjukkan data + mata kuliah)

5. **Buat request untuk Skenario 5** (Nested Resource)
   - Method: `GET`
   - URL: `{{base_url}}/api/students/245150707111012/mata-kuliah`
   - Klik "Send"
   - Capture screenshot (menunjukkan hanya array mata kuliah)

---

## KODE IMPLEMENTASI

### File: `app/Http/Controllers/StudentController.php`

**Validasi yang diperbarui:**
```php
// Store Method
$validated = $request->validate([
    'nim' => 'required|digits:15',
    'nama' => 'required|string|min:3|max:50',  // ✓ TAMBAHAN: min:3
    'mataKuliah' => 'required|array|min:1',    // ✓ TAMBAHAN: min:1
    'mataKuliah.*.kode' => 'required|regex:/^[A-Z]{3}[0-9]{5}$/',
    'mataKuliah.*.nama' => 'required|string|max:50',
    'mataKuliah.*.sks' => 'required|numeric|min:1|max:6',
]);

// Update Method
$validated = $request->validate([
    'nama' => 'sometimes|required|string|min:3|max:50',  // ✓ TAMBAHAN: min:3
    'mataKuliah' => 'sometimes|required|array|min:1',    // ✓ TAMBAHAN: min:1
    'mataKuliah.*.kode' => 'sometimes|required|regex:/^[A-Z]{3}[0-9]{5}$/',
    'mataKuliah.*.nama' => 'sometimes|required|string|max:50',
    'mataKuliah.*.sks' => 'sometimes|required|numeric|min:1|max:6',
]);
```

**Method Show (Compound Data):**
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

**Method Mata Kuliah By Student (Nested Resource):**
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

### File: `routes/api.php`

**Route yang diperbarui:**
```php
Route::get('/students/search', [StudentController::class, 'search']);
Route::get('/students', [StudentController::class, 'index']);
Route::post('/students', [StudentController::class, 'store']);
Route::get('/students/{nim}', [StudentController::class, 'show']);
Route::put('/students/{nim}', [StudentController::class, 'update']);
Route::patch('/students/{nim}', [StudentController::class, 'update']);
Route::delete('/students/{nim}', [StudentController::class, 'destroy']);
Route::get('/students/{nim}/mata-kuliah', [StudentController::class, 'mataKuliahByStudent']);
```

---

## ATURAN VALIDASI YANG DITERAPKAN

| Field | Rule | Keterangan |
|-------|------|-----------|
| `nim` | `required\|digits:15` | Wajib isi, tepat 15 digit |
| `nama` | `required\|string\|min:3\|max:50` | Wajib isi, string, min 3 kar, max 50 kar |
| `mataKuliah` | `required\|array\|min:1` | Wajib isi, harus array, minimal 1 item |
| `mataKuliah.*.kode` | `required\|regex:/^[A-Z]{3}[0-9]{5}$/` | Format: 3 huruf + 5 angka |
| `mataKuliah.*.nama` | `required\|string\|max:50` | Wajib isi, string, max 50 kar |
| `mataKuliah.*.sks` | `required\|numeric\|min:1\|max:6` | Wajib isi, angka, 1-6 |

---

## KESIMPULAN

Praktikum ini mengimplementasikan **REST API best practices** dengan:
1. ✓ Validasi yang lebih ketat dan meaningful
2. ✓ Response structure yang konsisten (message + data/errors)
3. ✓ Compound data untuk kebutuhan relasi lengkap
4. ✓ Nested resource untuk query yang spesifik dan focused
5. ✓ Error handling yang proper (422 untuk validation error)
6. ✓ RESTful endpoint design yang clean dan hierarchical
