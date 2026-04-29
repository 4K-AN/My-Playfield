# Panduan Testing JWT dengan Postman

## 1. REGISTER - POST /api/register

**URL:** `http://localhost:8000/api/register`

**Headers:**
```
Content-Type: application/json
Accept: application/json
```

**Body (raw JSON):**
```json
{
    "name": "Afirianto",
    "email": "afirianto@ub.ac.id",
    "password": "123456",
    "password_confirmation": "123456"
}
```

**Expected Response:**
```json
{
    "message": "User registered successfully (dummy)",
    "user": {
        "id": 421,
        "name": "Afirianto",
        "email": "afirianto@ub.ac.id",
        "password": "123456"
    }
}
```

---

## 2. LOGIN - POST /api/login

**URL:** `http://localhost:8000/api/login`

**Headers:**
```
Content-Type: application/json
Accept: application/json
```

**Body (raw JSON):**
```json
{
    "email": "user@example.com",
    "password": "password123"
}
```

**Expected Response:**
```json
{
    "message": "Login successful (dummy)",
    "token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
}
```

**⚠️ PENTING:** Copy token dari response, gunakan untuk request berikutnya!

---

## 3. PROFILE - GET /api/profile

**URL:** `http://localhost:8000/api/profile`

**Headers:**
```
Authorization: Bearer <JWT_TOKEN_DARI_LOGIN>
Accept: application/json
```

**Expected Response:**
```json
{
    "user": {
        "email": "user@example.com",
        "name": "User Cakep"
    }
}
```

---

## 4. LOGOUT - POST /api/logout

**URL:** `http://localhost:8000/api/logout`

**Headers:**
```
Authorization: Bearer <JWT_TOKEN_DARI_LOGIN>
Accept: application/json
```

**Expected Response:**
```json
{
    "message": "User logged out successfully"
}
```

---

## Testing Credentials (Dummy Users)

### User 1
- Email: `user@example.com`
- Password: `password123`
- Name: `User Cakep`

### User 2
- Email: `admin@example.com`
- Password: `secret321`
- Name: `Admin Hebat`

---

## Common Errors & Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| 401 Unauthorized | Token tidak ada atau expired | Login ulang dan ambil token baru |
| Token invalid or expired | Middleware tidak berjalan | Pastikan middleware terdaftar di bootstrap/app.php |
| Invalid email or password | Credentials salah | Gunakan email & password dari dummy users |
| SQLSTATE[HY000] | Database error | Abaikan jika menggunakan dummy data |

