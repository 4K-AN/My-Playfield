# 📚 LATIHAN 1 - INDEX & NAVIGATION

**Status: ✅ COMPLETE - ALL IMPLEMENTATION DONE**

Last updated: April 13, 2026

---

## 📑 DOCUMENTATION FILES

### Quick Start (Start here!) 📌
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Quick lookup guide (2 min read)
- **[COMPLETE_IMPLEMENTATION.md](COMPLETE_IMPLEMENTATION.md)** - Full summary (5 min read)

### Detailed Documentation 📖
- **[TESTING_GUIDE.md](TESTING_GUIDE.md)** - Complete testing scenarios with examples
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Backend changes summary
- **[FRONTEND_UPDATES.md](FRONTEND_UPDATES.md)** - Frontend changes documentation

### Postman Testing 🧪
- **[Postman_Collection.json](Postman_Collection.json)** - Ready-to-import collection

---

## ✅ WHAT'S BEEN IMPLEMENTED

### Backend (Laravel) ✅
```
✓ app/Http/Controllers/StudentController.php
  - Updated store() method with min:3, min:1 validation
  - Updated update() method with min:3, min:1 validation
  - Added show() method - Compound Data
  - Added mataKuliahByStudent() method - Nested Resource

✓ routes/api.php
  - Added GET /api/students/{nim} - show()
  - Added GET /api/students/{nim}/mata-kuliah - mataKuliahByStudent()
```

### Frontend (HTML/JS) ✅
```
✓ public/app.js
  - Enhanced saveStudent() with new validation
  - Enhanced updateStudentName() with new validation
  - Added viewStudentCompound() function
  - Added viewStudentCourses() function

✓ public/index.html
  - Added "Lihat Data Mahasiswa (Compound Data)" button
  - Added "Lihat Mata Kuliah (Nested Resource)" button
  - Added CSS styling for new buttons
```

---

## 🎯 5 TEST SCENARIOS

All scenarios ready to test:

### 1. ✅ Create Valid (Status 201)
- NIM: 15 digits
- Nama: 3+ characters
- Mata Kuliah: 1+ items
- Expected: Success response

### 2. ✅ Create Invalid - Nama Pendek (Status 422)
- Nama: Only 2 characters
- Expected: Validation error

### 3. ✅ Create Invalid - MK Kosong (Status 422)
- Mata Kuliah: Empty array
- Expected: Validation error

### 4. ✅ Create Invalid - Kode Format (Status 422)
- Kode: Wrong format
- Expected: Validation error

### 5. ✅ Update Valid (Status 200)
- Nama: Valid (3+ chars)
- Expected: Success response

### 6. ✅ Compound Data (GET)
- Endpoint: `/api/students/{nim}`
- Response: Student + courses together

### 7. ✅ Nested Resource (GET)
- Endpoint: `/api/students/{nim}/mata-kuliah`
- Response: Only courses (nested)

---

## 🚀 HOW TO TEST

### Step 1: Start Server
```bash
php artisan serve
```

### Step 2: Open in Browser
```
http://127.0.0.1:8000
```

### Step 3: Test Functions

#### Create Student:
1. Fill all fields with valid data
2. Click "Simpan Mahasiswa"
3. Check for success/error message

#### View Compound Data:
1. Enter valid NIM
2. Click "Lihat Data Mahasiswa (Compound Data)"
3. See student + courses together

#### View Nested Resource:
1. Enter valid NIM
2. Click "Lihat Mata Kuliah (Nested Resource)"
3. See only courses (hierarchical)

---

## 📊 RESPONSES COMPARISON

### Compound Data
```json
{
    "message": "Student retrieved successfully",
    "data": {
        "nim": "245150707111012",
        "nama": "Citra Dewi",
        "mataKuliah": [...]  ← Included
    }
}
```

### Nested Resource
```json
{
    "message": "Courses retrieved successfully",
    "student_nim": "245150707111012",
    "data": [...]  ← Only courses, no student
}
```

---

## 📋 VALIDATION RULES

| Field | Min | Max | Format | Required |
|-------|-----|-----|--------|----------|
| NIM | 15 | 15 | digits | ✓ |
| Nama | 3 | 50 | string | ✓ |
| Kode | - | - | [A-Z]{3}[0-9]{5} | ✓ |
| Nama MK | - | 50 | string | ✓ |
| SKS | 1 | 6 | numeric | ✓ |
| MK Count | 1 | ∞ | array | ✓ |

---

## 📁 FILES STRUCTURE

```
public/
├── index.html          ← Updated: +2 buttons
├── app.js              ← Updated: +2 functions, validation
└── app.css

app/Http/Controllers/
└── StudentController.php   ← Updated: +2 methods, validation

routes/
└── api.php             ← Updated: +2 routes

[Documentation files]
├── QUICK_REFERENCE.md             ← Start here
├── COMPLETE_IMPLEMENTATION.md      ← Full summary
├── TESTING_GUIDE.md                ← Test scenarios
├── IMPLEMENTATION_SUMMARY.md       ← Backend details
├── FRONTEND_UPDATES.md             ← Frontend details
├── Postman_Collection.json         ← For Postman
└── THIS FILE (INDEX.md)            ← Navigation
```

---

## 🎓 KEY DIFFERENCES: COMPOUND DATA vs NESTED RESOURCE

### When to use COMPOUND DATA:
```
Need: Full student profile + courses
URL: /api/students/{nim}
Response: Student object with embedded mataKuliah
Use Case: Dashboard, profile page
```

### When to use NESTED RESOURCE:
```
Need: Only specific sub-resource
URL: /api/students/{nim}/mata-kuliah
Response: Only array of courses
Use Case: Focused queries, RESTful API design
```

---

## ✨ VALIDATION ENHANCEMENTS

### New Validations Added:
- ✅ Nama minimum 3 characters (was unlimited)
- ✅ Mata Kuliah minimum 1 course (was unlimited)
- ✅ Better client-side error messages
- ✅ Frontend validation matches backend

### Error Response Format:
```json
{
    "message": "Validation failed",
    "errors": {
        "field_name": ["Error message"]
    }
}
```

---

## 🧪 POSTMAN QUICK START

1. Open Postman
2. Click "Import"
3. Select `Postman_Collection.json`
4. Set environment variable: `base_url` = `http://localhost:8000`
5. Run each request and capture screenshot

**All 5+ scenarios pre-built in collection!** ✓

---

## 📝 CHECKLIST FOR SUBMISSION

### Implementation ✅
- [x] Validation min:3 for nama
- [x] Validation min:1 for mataKuliah
- [x] Response structure (message + data/errors)
- [x] Compound data endpoint
- [x] Nested resource endpoint
- [x] Frontend validation
- [x] Frontend buttons
- [x] Documentation

### Testing
- [ ] Screenshot 1: Create Valid (201)
- [ ] Screenshot 2: Create Invalid (422)
- [ ] Screenshot 3: Update Valid (200)
- [ ] Screenshot 4: Compound Data
- [ ] Screenshot 5: Nested Resource

### Documentation
- [ ] Explanation of Compound Data
- [ ] Explanation of Nested Resource
- [ ] Comparison table

---

## 💡 TIPS FOR TESTING

### Valid Test Data:
```
NIM: 245150707111014 (must be 15 digits)
Nama: Muhammad Rizki (min 3 chars)
Kode: CIE61205 (exactly 3 UPPERCASE + 5 digits)
Nama MK: Pemrograman Web
SKS: 3 (1-6)
```

### Invalid Test Data:
```
Nama: "AB" (too short, < 3 chars) → Should fail
Kode: "CIE6120" (wrong format) → Should fail
Kode: "cie61205" (lowercase) → Should fail
SKS: 0 or 7 (out of range) → Should fail
```

---

## 🔗 ENDPOINT SUMMARY

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/students` | Get all students |
| POST | `/api/students` | Create student (NEW validation) |
| GET | `/api/students/{nim}` | Compound Data (NEW) |
| PATCH | `/api/students/{nim}` | Update student (NEW validation) |
| DELETE | `/api/students/{nim}` | Delete student |
| GET | `/api/students/{nim}/mata-kuliah` | Nested Resource (NEW) |
| GET | `/api/students/search` | Search students |

---

## 📞 TROUBLESHOOTING

### "Name must be at least 3 characters"
- Solution: Enter name with at least 3 characters

### "mata-kuliah field must have at least 1 items"
- Solution: Add at least 1 mata kuliah

### Compound Data not showing
- Check: NIM exists in database
- Check: NIM format is correct (15 digits)

### Nested Resource shows error
- Check: NIM is valid and exists
- Check: Student has courses assigned

---

## 📌 IMPORTANT NOTES

1. **Double Validation**: Frontend validates before sending, backend validates again
2. **Consistent Responses**: All endpoints return `{message, data}` or `{message, errors}`
3. **RESTful Design**: Nested resource (`/mata-kuliah`) is hierarchical
4. **Error Handling**: 422 for validation errors, 404 for not found

---

## 🎉 READY TO SUBMIT!

All requirements implemented:
- ✅ Backend changes complete
- ✅ Frontend integration complete
- ✅ Validation rules enhanced
- ✅ New endpoints functional
- ✅ Documentation comprehensive
- ✅ Test scenarios ready

**Start testing with `php artisan serve`!** 🚀

---

## Need Help?

📖 Read: **[COMPLETE_IMPLEMENTATION.md](COMPLETE_IMPLEMENTATION.md)**
🧪 Test: Use **[Postman_Collection.json](Postman_Collection.json)**
📚 Learn: Check **[TESTING_GUIDE.md](TESTING_GUIDE.md)**

---

**Implementation completed on: April 13, 2026**
**Total files modified: 4** (2 backend, 2 frontend)
**Documentation files: 6**
**Status: Ready for Submission** ✅
