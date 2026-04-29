# Checklist Implementasi JWT Authentication

## Tahap 1: Setup Awal
- [ ] Pastikan Laravel project berfungsi dengan baik
- [ ] Buka terminal di folder project
- [ ] Jalankan: `composer require tymonjwt-auth`
- [ ] Jalankan: `php artisan vendor:publish --provider=Tymon\JWTAuth\Providers\LaravelServiceProvider`
- [ ] Jalankan: `php artisan jwt:secret`
- [ ] Verifikasi file `.env` memiliki `JWT_SECRET=...`

## Tahap 2: Konfigurasi
- [ ] Buka `config/auth.php`
- [ ] Update bagian `'guards' => ['api' => ['driver' => 'jwt', ...]]`
- [ ] Simpan file

## Tahap 3: Membuat File-File
- [ ] Copy `DummyUser.php` ke `app/Models/DummyUser.php`
- [ ] Jalankan: `php artisan make:middleware DummyJwtMiddleware`
- [ ] Copy `DummyJwtMiddleware.php` ke `app/Http/Middleware/DummyJwtMiddleware.php`
- [ ] Jalankan: `php artisan make:controller Api/AuthController`
- [ ] Copy `AuthController.php` ke `app/Http/Controllers/Api/AuthController.php`

## Tahap 4: Register Middleware
- [ ] Buka `bootstrap/app.php`
- [ ] Add middleware alias dari `bootstrap_app_middleware.php` ke withMiddleware
```php
$middleware->alias([
    'dummy.jwt' => \App\Http\Middleware\DummyJwtMiddleware::class,
]);
```
- [ ] Simpan file

## Tahap 5: Update Routes
- [ ] Buka `routes/api.php`
- [ ] Copy & paste routes dari `routes_api.php`
- [ ] Verifikasi import controllers sudah benar
- [ ] Simpan file

## Tahap 6: Testing
- [ ] Jalankan Laravel: `php artisan serve`
- [ ] Buka Postman
- [ ] Test endpoint Register (POST /api/register)
- [ ] Test endpoint Login (POST /api/login) → **COPY TOKEN**
- [ ] Test endpoint Profile (GET /api/profile) dengan token
- [ ] Test endpoint Logout (POST /api/logout) dengan token

## Troubleshooting
- [ ] Jika error di middleware, pastikan path class benar
- [ ] Jika token invalid, regenerate dengan `php artisan jwt:secret`
- [ ] Jika 500 error, cek laravel log: `storage/logs/laravel.log`
- [ ] Jika database error, abaikan (menggunakan dummy data)

## Notes
- Dummy users sudah hardcoded di AuthController
- Tidak ada database yang terlibat (dummy only)
- Token valid untuk development testing
