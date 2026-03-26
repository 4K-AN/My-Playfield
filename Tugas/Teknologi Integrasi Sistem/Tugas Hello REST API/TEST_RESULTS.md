# API Search Feature Test Results

**Project:** TIS - Tugas Hello REST API (Student CRUD dengan Search)  
**Date:** March 26, 2026  
**Endpoint:** `GET /api/students/search`

---

## Implementation Summary

### Route Configuration
**File:** `routes/api.php`

```php
// Search route MUST be placed BEFORE dynamic parameter routes
Route::get('/students/search', [App\Http\Controllers\StudentController::class, 'search']);
Route::get('/students', [App\Http\Controllers\StudentController::class, 'index']);
Route::post('/students', [App\Http\Controllers\StudentController::class, 'store']);
Route::put('/students/{nim}', [App\Http\Controllers\StudentController::class, 'update']);
Route::delete('/students/{nim}', [App\Http\Controllers\StudentController::class, 'destroy']);
```

### Search Method Implementation
**File:** `app/Http/Controllers/StudentController.php`

The `search()` method:
- Captures query parameters: `nim`, `nama`, `kode_mk`
- Returns error 400 if no parameters provided
- Performs case-insensitive matching for nama using `stripos()`
- Exact matching for NIM
- Nested loop search for course codes in `mataKuliah` array
- Returns JSON response with message and results array

---

## Test Scenarios

### ✅ Scenario 1: Search by NIM (Exact Match)

**URL:** `http://127.0.0.1:8000/api/students/search?nim=245150707111012`

**Request:**
```
GET /api/students/search?nim=245150707111012
```

**Response Status:** 200 OK

**Response Body:**
```json
{
  "message": "Hasil pencarian",
  "data": [
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
    }
  ]
}
```

**Status:** ✅ PASSED - Returns 1 student matching exact NIM

---

### ✅ Scenario 2: Search by Nama (Case-Insensitive Contains)

**URL:** `http://127.0.0.1:8000/api/students/search?nama=Andy`

**Request:**
```
GET /api/students/search?nama=Andy
```

**Response Status:** 200 OK

**Response Body:**
```json
{
  "message": "Hasil pencarian",
  "data": [
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

**Status:** ✅ PASSED - Returns 1 student containing "Andy" in nama

---

### ✅ Scenario 3: Search by Kode MK (Course Code)

**URL:** `http://127.0.0.1:8000/api/students/search?kode_mk=CIE61205`

**Request:**
```
GET /api/students/search?kode_mk=CIE61205
```

**Response Status:** 200 OK

**Response Body:**
```json
{
  "message": "Hasil pencarian",
  "data": [
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

**Status:** ✅ PASSED - Returns 2 students enrolled in course CIE61205

---

### ✅ Scenario 4: No Parameters (Error Handling)

**URL:** `http://127.0.0.1:8000/api/students/search`

**Request:**
```
GET /api/students/search
```

**Response Status:** 400 Bad Request

**Response Body:**
```json
{
  "error": "Parameter tidak ditemukan. Harap masukkan nim, nama, atau kode_mk."
}
```

**Status:** ✅ PASSED - Returns 400 error with descriptive message when no parameters provided

---

## Test Summary

| Scenario | Test Case | URL | Expected | Actual | Status |
|----------|-----------|-----|----------|--------|--------|
| 1 | Search by NIM | `/api/students/search?nim=245150707111012` | 1 result (Citra Dewi) | 1 result (Citra Dewi) | ✅ PASS |
| 2 | Search by Nama | `/api/students/search?nama=Andy` | 1 result (Andy Lau) | 1 result (Andy Lau) | ✅ PASS |
| 3 | Search by Kode MK | `/api/students/search?kode_mk=CIE61205` | 2 results | 2 results | ✅ PASS |
| 4 | No Parameters | `/api/students/search` | Error 400 | Error 400 | ✅ PASS |

---

## Dummy Data Used

```php
$students = [
    [
        "nim" => "245150707111012",
        "nama" => "Citra Dewi",
        "mataKuliah" => [
            ["kode" => "CIE61205", "nama" => "PemWeb", "sks" => 3],
            ["kode" => "COM60015", "nama" => "MatDis", "sks" => 2]
        ]
    ],
    [
        "nim" => "245150707111013",
        "nama" => "Andy Lau",
        "mataKuliah" => [
            ["kode" => "CIE61205", "nama" => "PemWeb", "sks" => 3],
            ["kode" => "CIE61206", "nama" => "JarKom", "sks" => 3],
            ["kode" => "CIE61208", "nama" => "BasDat", "sks" => 3]
        ]
    ]
];
```

---

## Key Implementation Notes

1. **Route Placement:** Search route is placed BEFORE parameter-based routes to avoid route conflicts
2. **Parameter Handling:** Uses `$request->query()` to safely extract query parameters
3. **Validation:** Checks if at least one parameter is provided before searching
4. **Search Methods:**
   - NIM: Exact match comparison using `===`
   - Nama: Case-insensitive partial match using `stripos()`
   - Kode MK: Nested loop to check course codes in mataKuliah array
5. **Response Format:** Consistent JSON format with message and data array
6. **Error Status:** Returns 400 status code for missing parameters

---

## Files Modified

1. [routes/api.php](routes/api.php) - Added search route and student CRUD routes
2. [app/Http/Controllers/StudentController.php](app/Http/Controllers/StudentController.php) - Added search() method

---

**All tests completed successfully! ✅**
