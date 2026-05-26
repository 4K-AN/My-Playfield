# Supabase Backend Setup

## Cara Pakai

1. Buka **Supabase Dashboard** > **SQL Editor**.
2. Copy-paste seluruh isi file `001_schema.sql`.
3. Klik **Run** untuk mengeksekusi.

Script ini bersifat **idempotent** -- bisa dijalankan ulang tanpa error (menggunakan `IF NOT EXISTS`, `ON CONFLICT DO NOTHING`, dan `DROP TRIGGER IF EXISTS`).

## Yang Dibuat

### Tabel
| Tabel | Deskripsi |
|---|---|
| `profiles` | Profil user, 1:1 dengan `auth.users`. Otomatis dibuat saat signup via trigger. |
| `categories` | Kategori makanan global. Di-seed dengan 10 kategori default. |
| `food_items` | Entitas utama CRUD. Milik masing-masing user (`user_id`). |

### Row Level Security (RLS)
- **profiles**: User hanya bisa SELECT/UPDATE/INSERT profil miliknya sendiri.
- **categories**: Semua user yang login bisa SELECT.
- **food_items**: Full CRUD hanya untuk item milik user sendiri (`auth.uid() = user_id`).

### Triggers
- `on_auth_user_created`: Auto-insert ke `profiles` saat user baru sign up di Supabase Auth.
- `on_profiles_updated`: Auto-set `updated_at` ke `now()` saat profil di-update.

### Storage
- Bucket `food-images` (publik, max 5MB, JPEG/PNG/WebP/GIF).
- Upload path wajib menggunakan prefix `{user_id}/` agar sesuai RLS policy.
- Contoh path: `food-images/{user_id}/nasi-goreng.jpg`

### Seed Data (Kategori)
Makanan Berat, Makanan Ringan, Minuman, Dessert, Sarapan, Tradisional, Western, Asian, Seafood, Vegetarian.
