package com.example.fooddiary

import io.github.jan_tennert.supabase.createSupabaseClient
import io.github.jan_tennert.supabase.auth.Auth
import io.github.jan_tennert.supabase.postgrest.Postgrest
import io.github.jan_tennert.supabase.storage.Storage

object SupabaseClient {
    val client = createSupabaseClient(
        // Ganti dengan URL project Supabase kamu
        supabaseUrl = "[ISI_DENGAN_URL_KAMU_DI_SINI]",
        // Ganti dengan Anon Key project Supabase kamu
        supabaseKey = "[ISI_DENGAN_API_KEY_KAMU_DI_SINI]"
    ) {
        // Instalasi plugin untuk Database (Postgrest)
        install(Postgrest)
        
        // Instalasi plugin untuk Otentikasi (Auth)
        install(Auth)
        
        // Instalasi plugin untuk File Storage
        install(Storage)
    }
}
