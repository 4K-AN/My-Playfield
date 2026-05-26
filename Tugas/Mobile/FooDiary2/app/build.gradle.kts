plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.kotlin.serialization)
}

android {
    namespace = "com.example.foodiary2"
    compileSdk = 35 // Tetap gunakan 35 untuk stabilitas dengan AGP 8.7.3

    defaultConfig {
        applicationId = "com.example.foodiary2"
        minSdk = 24
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlinOptions {
        jvmTarget = "11"
    }
    buildFeatures {
        compose = true
    }
}

configurations.all {
    resolutionStrategy {
        // Paksa semua dependensi bermasalah ke versi stabil yang kompatibel dengan AGP 8.7.3
        force("androidx.core:core:1.15.0")
        force("androidx.core:core-ktx:1.15.0")
        force("androidx.activity:activity:1.10.0")
        force("androidx.activity:activity-compose:1.10.0")
        force("androidx.activity:activity-ktx:1.10.0")
        force("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")
        force("androidx.lifecycle:lifecycle-runtime-compose:2.8.7")
        force("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.7")
        force("androidx.browser:browser:1.8.0")
        
        // Mencegah penarikan otomatis library navigationevent yang butuh AGP tinggi
        eachDependency {
            if (requested.group == "androidx.navigationevent") {
                useVersion("1.0.0-alpha01") 
            }
        }
    }
}

dependencies {
    // UI & Core
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.activity.compose)
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.ui.graphics)
    implementation(libs.androidx.compose.ui.tooling.preview)
    implementation(libs.androidx.compose.material3)

    // Coil & Navigation
    implementation(libs.coil.compose)
    implementation(libs.androidx.navigation.compose)

    // Lifecycle
    implementation(libs.androidx.lifecycle.viewmodel.compose)
    implementation(libs.androidx.lifecycle.runtime.compose)

    // Supabase 3.x
    implementation(platform(libs.supabase.bom))
    implementation(libs.supabase.kt)
    implementation(libs.supabase.postgrest.kt)
    implementation(libs.supabase.auth.kt)
    implementation(libs.supabase.storage.kt)
    
    // Ktor (Wajib untuk Networking & JSON Supabase)
    implementation(libs.ktor.client.android)
    implementation(libs.ktor.client.content.negotiation)
    implementation(libs.ktor.serialization.kotlinx.json)

    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(platform(libs.androidx.compose.bom))
    androidTestImplementation(libs.androidx.compose.ui.test.junit4)
    debugImplementation(libs.androidx.compose.ui.tooling)
    debugImplementation(libs.androidx.compose.ui.test.manifest)
}
