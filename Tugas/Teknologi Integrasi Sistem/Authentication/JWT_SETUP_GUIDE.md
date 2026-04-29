# Setup JWT Authentication di Laravel - Panduan Lengkap

## Prasyarat
- Laravel project yang berfungsi dengan baik
- Composer terinstall
- PHP 8.0+

## Tahap-Tahap Implementasi

### BAGIAN 1: PERSIAPAN JWT
```bash
# Step 1: Masuk ke folder project
cd your-laravel-project

# Step 2: Install package JWT
composer require tymonjwt-auth

# Step 3: Publish konfigurasi package JWT
php artisan vendor:publish --provider=Tymon\JWTAuth\Providers\LaravelServiceProvider

# Step 4: Generate secret key JWT
php artisan jwt:secret
```

**Verifikasi**: Cek file `.env` memiliki `JWT_SECRET=...`

---

### BAGIAN 2: KONFIGURASI AUTH
Buka file `config/auth.php` dan ubah bagian `guards`:

```php
'guards' => [
    'web' => [
        'driver' => 'session',
        'provider' => 'users',
    ],
    'api' => [
        'driver' => 'jwt',
        'provider' => 'users',
    ],
],
```

---

### BAGIAN 3: MEMBUAT DUMMY USER
Buat file: `app/Models/DummyUser.php`

---

### BAGIAN 4: MEMBUAT MIDDLEWARE JWT KUSTOM
```bash
php artisan make:middleware DummyJwtMiddleware
```

Kemudian update file middleware dan register di `bootstrap/app.php`

---

### BAGIAN 5: MEMBUAT AUTHCONTROLLER
```bash
php artisan make:controller Api/AuthController
```

---

### BAGIAN 6: MENAMBAHKAN ROUTE API
Update file `routes/api.php` dengan routes autentikasi

---

### BAGIAN 7: UJI COBA DI POSTMAN
Test semua endpoint: register, login, profile, logout

---

## Endpoint API
- **POST** `/api/register` - Daftar user baru
- **POST** `/api/login` - Login dan dapatkan token
- **GET** `/api/profile` - Lihat profil (memerlukan token)
- **POST** `/api/logout` - Logout (memerlukan token)
