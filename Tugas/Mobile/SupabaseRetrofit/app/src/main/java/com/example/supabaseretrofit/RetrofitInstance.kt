package com.example.supabaseretrofit

import okhttp3.OkHttpClient
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory

object RetrofitInstance {
    private const val BASE_URL = "https://[project_id].supabase.co/rest/v1/"

    private val retrofit: Retrofit by lazy {
        val sbHttpClient = OkHttpClient().newBuilder()
            .addInterceptor(ApiKeyInterceptor("[publishable key]"))
            .build()

        Retrofit.Builder()
            .baseUrl(BASE_URL)
            .client(sbHttpClient)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
    }

    val bookService: BookService by lazy {
        retrofit.create(BookService::class.java)
    }
}
