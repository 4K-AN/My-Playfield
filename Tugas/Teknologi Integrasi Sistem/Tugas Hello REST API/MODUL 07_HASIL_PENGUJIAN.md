# Dokumentasi Pengujian Authorization Berbasis Role pada JWT

Berikut ini adalah hasil dari pengujian skenario yang diminta pada Tugas Praktikum:

## 1. User biasa gagal mengakses admin dashboard
- **Role:** user
- **Endpoint:** `GET /api/admin/dashboard`
- **Status Code:** 403 Forbidden
- **Response JSON:**
```json
{
  "message": "Access denied. You do not have the required role."
}
```

## 2. User biasa berhasil mengakses user dashboard
- **Role:** user
- **Endpoint:** `GET /api/user/dashboard`
- **Status Code:** 200 OK
- **Response JSON:**
```json
{
  "message": "Welcome to User Dashboard"
}
```

## 3. Admin berhasil mengakses admin dashboard
- **Role:** admin
- **Endpoint:** `GET /api/admin/dashboard`
- **Status Code:** 200 OK
- **Response JSON:**
```json
{
  "message": "Welcome to Admin Dashboard"
}
```

## 4. Admin gagal mengakses user dashboard
- **Role:** admin
- **Endpoint:** `GET /api/user/dashboard`
- **Status Code:** 403 Forbidden
- **Response JSON:**
```json
{
  "message": "Access denied. You do not have the required role."
}
```

## 5. Manager berhasil mengakses manager dashboard
- **Role:** manager
- **Endpoint:** `GET /api/manager/dashboard`
- **Status Code:** 200 OK
- **Response JSON:**
```json
{
  "message": "Welcome to Manager Dashboard"
}
```

## 6. Manager gagal mengakses admin dashboard (Skenario Tambahan)
- **Role:** manager
- **Endpoint:** `GET /api/admin/dashboard`
- **Status Code:** 403 Forbidden
- **Response JSON:**
```json
{
  "message": "Access denied. You do not have the required role."
}
```
