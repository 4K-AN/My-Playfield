package com.example.foodiary2

import io.github.jan.supabase.createSupabaseClient
import io.github.jan.supabase.auth.Auth
import io.github.jan.supabase.postgrest.Postgrest
import io.github.jan.supabase.storage.Storage

object SupabaseClient {
    val client = createSupabaseClient(
        // Ganti dengan URL project Supabase kamu
        supabaseUrl = "https://fleibijqudqxwzsjipxy.supabase.co",
        // Ganti dengan Anon Key project Supabase kamu
        supabaseKey = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZsZWliaWpxdWRxeHd6c2ppcHh5Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzgwOTgxNjAsImV4cCI6MjA5MzY3NDE2MH0.1ssZt67IR3nwrj769h5IMU9UB8VVombmdVg_Po0oohY"
    ) {
        install(Postgrest)
        install(Auth)
        install(Storage)
    }
}
