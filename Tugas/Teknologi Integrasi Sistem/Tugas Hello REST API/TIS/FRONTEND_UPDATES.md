# Frontend Updates - Latihan 1

## Summary of Changes

Frontend (`public/app.js` dan `public/index.html`) telah diperbarui untuk mendukung validasi baru dan endpoint baru.

---

## 1. ✅ Enhanced Client-Side Validation

### File: `public/app.js`

#### a) `saveStudent()` function - Updated with NEW validations:
- ✅ NIM: must be exactly 15 digits
- ✅ Name: minimum 3 characters, maximum 50 characters (NEW)
- ✅ Course code format: 3 uppercase letters + 5 digits (e.g., CIE61205)
- ✅ SKS: numeric, between 1-6
- ✅ All fields required

**Error Messages untuk User:**
```
"Nama harus minimal 3 karakter!"
"Kode mata kuliah harus format 3 huruf kapital + 5 angka (contoh: CIE61205)!"
"SKS harus antara 1-6!"
```

#### b) `updateStudentName()` function - Updated:
- ✅ NIM validation: 15 digits
- ✅ Name validation: minimum 3 characters (NEW)
- ✅ Better error messages

---

## 2. ✅ TWO NEW FUNCTIONS ADDED

### Function 1: `viewStudentCompound()`
**Purpose:** Retrieve single student with all their courses (Compound Data)

**HTTP Request:**
```
GET /api/students/{nim}
```

**Response Format:**
```json
{
    "message": "Student retrieved successfully",
    "data": {
        "nim": "245150707111012",
        "nama": "Citra Dewi",
        "mataKuliah": [...]
    }
}
```

**Usage:** Click button "Lihat Data Mahasiswa (Compound Data)"

---

### Function 2: `viewStudentCourses()`
**Purpose:** Retrieve ONLY courses for a student (Nested Resource)

**HTTP Request:**
```
GET /api/students/{nim}/mata-kuliah
```

**Response Format:**
```json
{
    "message": "Courses retrieved successfully",
    "student_nim": "245150707111012",
    "data": [...]  // Only array of courses
}
```

**Usage:** Click button "Lihat Mata Kuliah (Nested Resource)"

---

## 3. ✅ NEW BUTTONS ADDED TO UI

### File: `public/index.html`

#### Compound Data Button:
```html
<button id="btn-compound" onclick="viewStudentCompound()">
    Lihat Data Mahasiswa (Compound Data)
</button>
```
- Color: Cyan (#17a2b8)
- Fetches: Student profile + courses together

#### Nested Resource Button:
```html
<button id="btn-nested" onclick="viewStudentCourses()">
    Lihat Mata Kuliah (Nested Resource)
</button>
```
- Color: Purple (#6f42c1)
- Fetches: Only courses for the student

---

## 4. ✅ VALIDATION RULES REFERENCE

### Create/Update Student - Name Validation:

```javascript
// Minimum 3 characters
if (nama.length < 3) {
    errorDiv.innerHTML = 'Nama harus minimal 3 karakter!';
    return;
}

// Maximum 50 characters
if (nama.length > 50) {
    errorDiv.innerHTML = 'Nama maksimal 50 karakter!';
    return;
}
```

### Create Student - Course Code Format:

```javascript
// Format: 3 uppercase letters + 5 digits
const kodeRegex = /^[A-Z]{3}[0-9]{5}$/;
if (!kodeRegex.test(kode)) {
    errorDiv.innerHTML = 'Kode harus format 3 huruf kapital + 5 angka!';
    return;
}
```

---

## 5. User Interface Changes

### Before:
```
Button Group 1:
- Tampilkan Semua
- Simpan Mahasiswa
- Ubah Nama (PATCH)
- Hapus Mahasiswa
```

### After:
```
Button Group 1 (Original):
- Tampilkan Semua
- Simpan Mahasiswa
- Ubah Nama (PATCH)
- Hapus Mahasiswa

Button Group 2 (NEW):
- Lihat Data Mahasiswa (Compound Data)    [Cyan]
- Lihat Mata Kuliah (Nested Resource)     [Purple]
```

---

## 6. Error Handling Improvements

### Form Validation Errors (Before Submit):
```
"Semua field harus diisi!"
"NIM harus terdiri dari 15 digit angka!"
"Nama harus minimal 3 karakter!"
"Nama maksimal 50 karakter!"
"Kode harus format 3 huruf kapital + 5 angka!"
"SKS harus antara 1-6!"
```

### Server Validation Errors (After Submit):
```
If server validation fails:
{
    "message": "Validation failed",
    "errors": {
        "nama": ["The nama field must be at least 3 characters."]
    }
}
```

Frontend displays these server errors in the error div.

---

## 7. Testing Checklist

- [ ] Test Create with valid name (3+ characters)
- [ ] Test Create with invalid name (2 characters) - Should show client error
- [ ] Test Create with invalid course code format - Should show client error
- [ ] Test Update name with valid name (3+ characters)
- [ ] Test Update name with invalid name (2 characters) - Should show error
- [ ] Test "Lihat Data Mahasiswa (Compound Data)" - Shows student + courses
- [ ] Test "Lihat Mata Kuliah (Nested Resource)" - Shows only courses
- [ ] Test with non-existent NIM - Should show 404 error

---

## 8. How It Works Now

### Flow 1: Create Student (with new validation)
1. User fills form: NIM, Nama, Kode, Nama MK, SKS
2. User clicks "Simpan Mahasiswa"
3. Frontend validates:
   - NIM: 15 digits? ✓
   - Nama: 3+ characters? ✓
   - Kode: Format AAA##### ? ✓
   - All fields filled? ✓
4. If all valid → Send to backend
5. Backend validates again (double validation)
6. Success/Error response displayed

### Flow 2: View Compound Data
1. User enters NIM
2. User clicks "Lihat Data Mahasiswa (Compound Data)"
3. Sends: `GET /api/students/{nim}`
4. Response: Student object with mataKuliah array embedded
5. Display entire student card with courses

### Flow 3: View Nested Resource
1. User enters NIM
2. User clicks "Lihat Mata Kuliah (Nested Resource)"
3. Sends: `GET /api/students/{nim}/mata-kuliah`
4. Response: Only array of courses
5. Display courses focused list

---

## 9. Files Modified

| File | Changes |
|------|---------|
| `public/app.js` | Added 2 new functions + enhanced validation |
| `public/index.html` | Added 2 new buttons + CSS for new buttons |

---

## 10. Ready for Testing ✅

Frontend is now fully synchronized with the backend API:
- ✅ Validation matches backend requirements
- ✅ New endpoints integrated
- ✅ New buttons added to UI
- ✅ Error handling improved
- ✅ Support for Compound Data
- ✅ Support for Nested Resource

**Start Laravel server and test!**
```bash
php artisan serve
```

Then open browser and test the new buttons! 🎉
