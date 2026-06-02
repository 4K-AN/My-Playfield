plugins {
    alias(libs.plugins.androidApplication)
    alias(libs.plugins.kotlinAndroid)
    alias(libs.plugins.kotlinCompose)
    alias(libs.plugins.kotlinSerialization)
}

android {
    namespace = "com.example.foodiary2"
    compileSdk = 35 // Menggunakan SDK 35 untuk stabilitas dengan AGP 8.7.3

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
    buildFeatures {
        compose = true
    }
}

// Kotlin 2.3+ compiler options (migrated from deprecated kotlinOptions.jvmTarget)
kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_11)
    }
}

// FORCE LOCK: Memastikan library tidak naik ke versi yang butuh AGP 8.9.1+
configurations.all {
    resolutionStrategy {
        force("androidx.core:core:1.15.0")
        force("androidx.core:core-ktx:1.15.0")
        force("androidx.activity:activity:1.10.0")
        force("androidx.activity:activity-compose:1.10.0")
        force("androidx.activity:activity-ktx:1.10.0")
        force("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")
        force("androidx.lifecycle:lifecycle-runtime-compose:2.8.7")
        force("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.7")
        force("androidx.browser:browser:1.8.0")
    }
}

dependencies {
    // UI & Core (Menggunakan camelCase alias sesuai libs.versions.toml)
    implementation(libs.androidxCoreKtx)
    implementation(libs.androidxLifecycleRuntimeKtx)
    implementation(libs.androidxActivityCompose)
    implementation(platform(libs.androidxComposeBom))
    implementation(libs.androidxComposeUi)
    implementation(libs.androidxComposeUiGraphics)
    implementation(libs.androidxComposeUiToolingPreview)
    implementation(libs.androidxComposeMaterial3)

    // Coil & Navigation
    implementation(libs.coilCompose)
    implementation(libs.androidxNavigationCompose)

    // Lifecycle
    implementation(libs.androidxLifecycleViewmodelCompose)
    implementation(libs.androidxLifecycleRuntimeCompose)

    // Supabase 3.x
    implementation(platform(libs.supabaseBom))
    implementation(libs.supabaseKt)
    implementation(libs.supabasePostgrest)
    implementation(libs.supabaseAuth)
    implementation(libs.supabaseStorage)
    
    // Ktor
    implementation(libs.ktorClientAndroid)
    implementation(libs.ktorClientContentNegotiation)
    implementation(libs.ktorSerializationJson)

    testImplementation(libs.junit)
    androidTestImplementation(libs.androidxJunit)
    androidTestImplementation(libs.androidxEspressoCore)
    androidTestImplementation(platform(libs.androidxComposeBom))
    androidTestImplementation(libs.androidxComposeUiTestJunit4)
    debugImplementation(libs.androidxComposeUiTooling)
    debugImplementation(libs.androidxComposeUiTestManifest)
}
