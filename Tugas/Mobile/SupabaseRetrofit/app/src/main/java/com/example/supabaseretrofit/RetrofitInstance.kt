package com.example.supabaseretrofit

import okhttp3.OkHttpClient
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory

object

RetrofitInstance {
    private const val BASE_URL = "https://kepbltghhklkkldkbcjl.supabase.co/rest/v1/"

    private val retrofit: Retrofit by lazy {
        val sbHttpClient = OkHttpClient().newBuilder()
            .addInterceptor(ApiKeyInterceptor("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImtlcGJsdGdoaGtsa2tsZGtiY2psIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Nzk1NjcxMTgsImV4cCI6MjA5NTE0MzExOH0.epd9elj1mxmUl2cn4Y_8GL4TfoXG6tXcNqwn5Id-r5Q"))
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
