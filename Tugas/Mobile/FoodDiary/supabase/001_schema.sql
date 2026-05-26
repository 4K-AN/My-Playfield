-- ============================================================
-- Food Diary - Supabase Backend Setup
-- ============================================================
-- Jalankan script ini di Supabase SQL Editor (Dashboard > SQL Editor)
-- secara berurutan dari atas ke bawah.
-- ============================================================


-- ============================================================
-- 1. EXTENSIONS
-- ============================================================
CREATE EXTENSION IF NOT EXISTS "pgcrypto";


-- ============================================================
-- 2. TABEL: profiles
--    Relasi 1:1 dengan auth.users. Dibuat otomatis via trigger.
-- ============================================================
CREATE TABLE IF NOT EXISTS public.profiles (
    id          UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    full_name   VARCHAR(255),
    avatar_url  TEXT,
    updated_at  TIMESTAMPTZ DEFAULT now()
);

COMMENT ON TABLE public.profiles IS 'Profil pengguna, 1:1 dengan auth.users.';


-- ============================================================
-- 3. TABEL: categories
--    Kategori global (tidak per-user).
-- ============================================================
CREATE TABLE IF NOT EXISTS public.categories (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        VARCHAR(100) NOT NULL UNIQUE,
    created_at  TIMESTAMPTZ DEFAULT now()
);

COMMENT ON TABLE public.categories IS 'Kategori makanan global.';


-- ============================================================
-- 4. TABEL: food_items
--    Entitas utama CRUD, milik masing-masing user.
-- ============================================================
CREATE TABLE IF NOT EXISTS public.food_items (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    category_id UUID REFERENCES public.categories(id) ON DELETE SET NULL,
    title       VARCHAR(255) NOT NULL,
    description TEXT,
    image_url   TEXT,
    created_at  TIMESTAMPTZ DEFAULT now()
);

COMMENT ON TABLE public.food_items IS 'Item makanan milik user.';

-- Index untuk query per-user
CREATE INDEX IF NOT EXISTS idx_food_items_user_id ON public.food_items(user_id);
-- Index untuk filter per-kategori
CREATE INDEX IF NOT EXISTS idx_food_items_category_id ON public.food_items(category_id);


-- ============================================================
-- 5. ROW LEVEL SECURITY (RLS)
-- ============================================================

-- ---- profiles ----
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;

-- User hanya bisa melihat profilnya sendiri
CREATE POLICY "profiles_select_own"
    ON public.profiles FOR SELECT
    USING (auth.uid() = id);

-- User hanya bisa mengupdate profilnya sendiri
CREATE POLICY "profiles_update_own"
    ON public.profiles FOR UPDATE
    USING (auth.uid() = id)
    WITH CHECK (auth.uid() = id);

-- Insert ditangani oleh trigger (service_role), tapi user juga boleh insert profilnya sendiri
CREATE POLICY "profiles_insert_own"
    ON public.profiles FOR INSERT
    WITH CHECK (auth.uid() = id);


-- ---- categories ----
ALTER TABLE public.categories ENABLE ROW LEVEL SECURITY;

-- Semua user yang sudah login bisa melihat kategori
CREATE POLICY "categories_select_authenticated"
    ON public.categories FOR SELECT
    USING (auth.role() = 'authenticated');


-- ---- food_items ----
ALTER TABLE public.food_items ENABLE ROW LEVEL SECURITY;

-- User hanya bisa melihat item miliknya
CREATE POLICY "food_items_select_own"
    ON public.food_items FOR SELECT
    USING (auth.uid() = user_id);

-- User hanya bisa insert item untuk dirinya sendiri
CREATE POLICY "food_items_insert_own"
    ON public.food_items FOR INSERT
    WITH CHECK (auth.uid() = user_id);

-- User hanya bisa update item miliknya
CREATE POLICY "food_items_update_own"
    ON public.food_items FOR UPDATE
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

-- User hanya bisa delete item miliknya
CREATE POLICY "food_items_delete_own"
    ON public.food_items FOR DELETE
    USING (auth.uid() = user_id);


-- ============================================================
-- 6. TRIGGER: Auto-create profile saat user sign up
-- ============================================================
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
BEGIN
    INSERT INTO public.profiles (id, full_name, avatar_url)
    VALUES (
        NEW.id,
        COALESCE(NEW.raw_user_meta_data ->> 'full_name', ''),
        COALESCE(NEW.raw_user_meta_data ->> 'avatar_url', '')
    );
    RETURN NEW;
END;
$$;

-- Drop trigger dulu kalau sudah ada (idempotent)
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;

CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW
    EXECUTE FUNCTION public.handle_new_user();


-- ============================================================
-- 7. TRIGGER: Auto-update updated_at pada profiles
-- ============================================================
CREATE OR REPLACE FUNCTION public.handle_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS on_profiles_updated ON public.profiles;

CREATE TRIGGER on_profiles_updated
    BEFORE UPDATE ON public.profiles
    FOR EACH ROW
    EXECUTE FUNCTION public.handle_updated_at();


-- ============================================================
-- 8. STORAGE: Bucket food-images
-- ============================================================
-- Buat bucket publik untuk gambar makanan.
-- Supabase Storage bucket creation via SQL:
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
    'food-images',
    'food-images',
    true,                                       -- publik agar bisa diakses via public URL
    5242880,                                    -- 5 MB limit per file
    ARRAY['image/jpeg', 'image/png', 'image/webp', 'image/gif']
)
ON CONFLICT (id) DO NOTHING;


-- ---- Storage RLS Policies ----

-- User yang login bisa upload ke folder miliknya sendiri (path: user_id/*)
CREATE POLICY "storage_food_images_insert"
    ON storage.objects FOR INSERT
    WITH CHECK (
        bucket_id = 'food-images'
        AND auth.role() = 'authenticated'
        AND (storage.foldername(name))[1] = auth.uid()::text
    );

-- User yang login bisa update file miliknya sendiri
CREATE POLICY "storage_food_images_update"
    ON storage.objects FOR UPDATE
    USING (
        bucket_id = 'food-images'
        AND auth.role() = 'authenticated'
        AND (storage.foldername(name))[1] = auth.uid()::text
    );

-- User yang login bisa delete file miliknya sendiri
CREATE POLICY "storage_food_images_delete"
    ON storage.objects FOR DELETE
    USING (
        bucket_id = 'food-images'
        AND auth.role() = 'authenticated'
        AND (storage.foldername(name))[1] = auth.uid()::text
    );

-- Siapa saja bisa melihat (karena bucket publik)
CREATE POLICY "storage_food_images_select"
    ON storage.objects FOR SELECT
    USING (bucket_id = 'food-images');


-- ============================================================
-- 9. SEED DATA: Kategori default
-- ============================================================
INSERT INTO public.categories (name) VALUES
    ('Makanan Berat'),
    ('Makanan Ringan'),
    ('Minuman'),
    ('Dessert'),
    ('Sarapan'),
    ('Tradisional'),
    ('Western'),
    ('Asian'),
    ('Seafood'),
    ('Vegetarian')
ON CONFLICT (name) DO NOTHING;
