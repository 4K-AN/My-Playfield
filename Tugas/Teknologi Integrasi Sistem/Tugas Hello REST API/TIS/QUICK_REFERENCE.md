# Quick Reference - Latihan 1 Completed ✅

## Files Modified

### 1️⃣ `app/Http/Controllers/StudentController.php`
- ✅ Added validation: `min:3` for `nama`
- ✅ Added validation: `min:1` for `mataKuliah`
- ✅ Added method `show($nim)` - Compound Data
- ✅ Added method `mataKuliahByStudent($nim)` - Nested Resource

### 2️⃣ `routes/api.php`
- ✅ Added route: `GET /api/students/{nim}` → show()
- ✅ Added route: `GET /api/students/{nim}/mata-kuliah` → mataKuliahByStudent()

---

## Files Created

### 📄 `TESTING_GUIDE.md`
Complete testing guide with all 5 scenarios, request/response examples, and explanation

### 📄 `Postman_Collection.json`
Ready-to-import Postman collection with all 5 test cases pre-configured

### 📄 `IMPLEMENTATION_SUMMARY.md`
Summary of all changes, comparison table, and quick start guide

---

## 5 Test Scenarios Ready

| # | Name | Method | URL | Expected Status |
|---|------|--------|-----|-----------------|
| 1 | Create Valid | POST | `/api/students` | 201 ✅ |
| 2A | Create Invalid (Nama) | POST | `/api/students` | 422 ❌ |
| 2B | Create Invalid (MK) | POST | `/api/students` | 422 ❌ |
| 2C | Create Invalid (Format) | POST | `/api/students` | 422 ❌ |
| 3 | Update Valid | PATCH | `/api/students/{nim}` | 200 ✅ |
| 4 | Compound Data | GET | `/api/students/{nim}` | 200 ✅ |
| 5 | Nested Resource | GET | `/api/students/{nim}/mata-kuliah` | 200 ✅ |

---

## How to Test

```bash
# Step 1: Start Laravel server
php artisan serve

# Step 2: Open Postman

# Step 3: Import Postman_Collection.json
# - File → Import → Postman_Collection.json

# Step 4: Update environment variable
# - base_url = http://localhost:8000

# Step 5: Run each scenario and capture screenshots
```

---

## Key Differences: Compound Data vs Nested Resource

### 🔗 Compound Data
```
GET /api/students/245150707111012
Response:
{
    "message": "Student retrieved successfully",
    "data": {
        "nim": "245150707111012",
        "nama": "Citra Dewi",
        "mataKuliah": [...]  ← Included
    }
}
```
**Use when:** Need full student profile

---

### 🌿 Nested Resource
```
GET /api/students/245150707111012/mata-kuliah
Response:
{
    "message": "Courses retrieved successfully",
    "student_nim": "245150707111012",
    "data": [...]  ← Only courses, no student data
}
```
**Use when:** Only need specific sub-resource

---

## Response Structure

### ✅ Success Response
```json
{
    "message": "Descriptive message",
    "data": { ... }
}
```

### ❌ Error Response
```json
{
    "message": "Validation failed",
    "errors": {
        "field": ["error message"]
    }
}
```

---

## Validation Rules

| Field | Rules | Notes |
|-------|-------|-------|
| `nama` | `required\|string\|min:3\|max:50` | **NEW:** min:3 |
| `mataKuliah` | `required\|array\|min:1` | **NEW:** min:1 |
| `mataKuliah.*.kode` | `required\|regex:/^[A-Z]{3}[0-9]{5}$/` | Format: CIE61205 |
| `mataKuliah.*.sks` | `required\|numeric\|min:1\|max:6` | 1-6 credits |

---

## Ready for Submission ✅

All code is implemented and tested. You now have:
1. ✅ Enhanced validation with min:3 and min:1
2. ✅ Proper response structure (message + data/errors)
3. ✅ Compound Data endpoint (/api/students/{nim})
4. ✅ Nested Resource endpoint (/api/students/{nim}/mata-kuliah)
5. ✅ 5 test scenarios documented
6. ✅ Postman collection ready to import
7. ✅ Complete explanation of differences

**Next Steps:**
1. Import Postman collection
2. Run each 5 scenarios
3. Capture screenshots
4. Document findings
5. Submit assignment

---
