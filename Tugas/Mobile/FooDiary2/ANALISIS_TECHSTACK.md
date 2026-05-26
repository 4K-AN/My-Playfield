# Analisis Tech Stack — FooDiary(2)

## Gambaran Umum

**FooDiary(2)** adalah proyek Android yang baru dibuat melalui template **Android Studio > New Project > Empty Compose Activity**. Proyek ini masih dalam tahap paling awal (_scaffolding_) dan **belum memiliki fitur bisnis apapun** — tidak ada fungsionalitas diary, food tracker, atau koneksi API. Nama "FooDiary(2)" bersifat aspirasional; seluruh kode saat ini hanya menampilkan teks "Hello Android!".

---

## Ringkasan Tech Stack

| Komponen | Teknologi | Versi |
|---|---|---|
| **Bahasa** | Kotlin | 2.0.21 |
| **UI Toolkit** | Jetpack Compose + Material3 | Compose BOM 2024.09.00 |
| **Build System** | Gradle + Android Gradle Plugin (AGP) | Gradle 8.13 / AGP 8.13.2 |
| **Min / Target / Compile SDK** | Android API | 24 / 36 / 36 |
| **JVM Target** | Java | 11 |
| **Dependency Management** | Version Catalog (`libs.versions.toml`) | — |
| **Compose Compiler** | Plugin `kotlin.plugin.compose` (built-in) | — |
| **Test Runner** | AndroidJUnitRunner | — |

---

## Daftar Dependency (Version Catalog)

### Production Dependencies

| Library | Artifact | Versi | Fungsi |
|---|---|---|---|
| **Core KTX** | `androidx.core:core-ktx` | 1.18.0 | Ekstensi Kotlin untuk framework Android |
| **Lifecycle Runtime KTX** | `androidx.lifecycle:lifecycle-runtime-ktx` | 2.10.0 | Lifecycle-aware components + Coroutine scope |
| **Activity Compose** | `androidx.activity:activity-compose` | 1.13.0 | Integrasi Activity dengan Compose |
| **Compose BOM** | `androidx.compose:compose-bom` | 2024.09.00 | Bill of Materials untuk semua library Compose |
| **Compose UI** | `androidx.compose.ui:ui` | (via BOM) | Foundation Compose UI |
| **Compose UI Graphics** | `androidx.compose.ui:ui-graphics` | (via BOM) | Canvas, rendering, brush |
| **Compose UI Tooling Preview** | `androidx.compose.ui:ui-tooling-preview` | (via BOM) | Preview di Android Studio |
| **Material3** | `androidx.compose.material3:material3` | (via BOM) | Material Design 3 components |

### Debug Dependencies
- `androidx.compose.ui:ui-tooling` — Tooling untuk debugging composable
- `androidx.compose.ui:ui-test-manifest` — Manifest untuk instrumentation test

### Test Dependencies
- `junit:junit:4.13.2` — Unit testing
- `androidx.test.ext:junit:1.3.0` — AndroidX JUnit extensions
- `androidx.compose.ui:ui-test-junit4` — Compose UI testing
- `androidx.test.espresso:espresso-core:3.7.0` — Espresso UI testing

---

## Yang TIDAK Ada (Absen dari Proyek)

Library/library yang **tidak digunakan** sama sekali di proyek ini:

| Library | Keterangan |
|---|---|
| **Room** | Tidak ada database lokal |
| **Retrofit / OkHttp / Ktor** | Tidak ada networking / HTTP client |
| **Supabase Client** | Tidak ada backend |
| **Hilt / Koin / Dagger** | Tidak ada Dependency Injection |
| **Navigation Compose** | Tidak ada multi-screen / navigasi |
| **ViewModel / LiveData / Flow** | Tidak ada state management |
| **Coroutines** (eksplisit) | Tidak ada penggunaan async programming |
| **DataStore / SharedPreferences** | Tidak ada penyimpanan preferensi |
| **Coil / Glide** | Tidak ada image loading |
| **Kotlin Serialization / Parcelize** | Tidak ada serialization plugin |
| **Application Subclass** | Tidak ada kustomisasi Application |

---

## Arsitektur

**Belum ada arsitektur yang diterapkan.** Proyek masih menggunakan pola default template:

```
MainActivity (ComponentActivity)
  └─ setContent {
        FooDiary2Theme {
            Scaffold {
                Greeting("Android")   // → Text("Hello Android!")
            }
        }
     }
```

- **No MVVM / MVI / MVC** — hanya satu Activity dengan satu Composable function
- **No Data Layer** — tidak ada repository, DAO, atau API service
- **No Navigation** — hanya satu screen statis
- **No DI** — semua object dibuat manual (tidak ada)

---

## Struktur Direktori (Source Code)

```
app/src/main/java/com/example/foodiary2/
├── MainActivity.kt           # Entry point: ComponentActivity + setContent
└── ui/
    └── theme/
        ├── Color.kt           # Palet warna (Purple/Pink untuk light & dark)
        ├── Theme.kt           # FooDiary2Theme dengan dynamic color support
        └── Type.kt            # Typography (hanya bodyLarge)
```

Total **4 file Kotlin** (~150 LOC) untuk kode utama.

---

## Detail Konfigurasi Build

### Root `build.gradle.kts`
Hanya mendeklarasikan plugin (tanpa apply):
- `com.android.application` — AGP
- `org.jetbrains.kotlin.android` — Kotlin Android
- `org.jetbrains.kotlin.plugin.compose` — Compose compiler plugin

### Module `app/build.gradle.kts`
- Mengaktifkan Compose via `buildFeatures { compose = true }`
- Compose compiler diatur oleh plugin Kotlin (tidak perlu `kotlinCompilerExtensionVersion`)
- Semua dependency dari Version Catalog (`libs.versions.toml`)

### `gradle.properties`
- `android.useAndroidX = true`
- `kotlin.code.style = official`
- `android.enableR8.fullMode = true`
- `org.gradle.jvmargs` dengan alokasi heap dan locale

---

## Kesimpulan

| Aspek | Status |
|---|---|
| **Kesiapan Produksi** | ❌ — Baru template kosong |
| **Fungsionalitas** | Tidak ada (hanya Hello World) |
| **Potensi Pengembangan** | Tinggi — fondasi Compose + Material3 sudah siap |
| **Yang Paling Dibutuhkan** | ViewModel, Navigation, Data Layer (Room/Retrofit/Supabase), DI |

Proyek ini adalah **blank canvas** — tinggal memilih tech stack tambahan (misalnya Supabase seperti proyek FoodDiary sebelumnya, atau Room untuk lokal) untuk membangun fitur food diary yang sesungguhnya.
