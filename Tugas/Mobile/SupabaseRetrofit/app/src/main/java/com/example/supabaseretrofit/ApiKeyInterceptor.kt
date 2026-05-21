package com.example.supabaseretrofit

import okhttp3.Interceptor
import okhttp3.Response

class ApiKeyInterceptor(private val apikey: String) : Interceptor {
    override fun intercept(chain: Interceptor.Chain): Response {
        val originalRequest = chain.request()
        val requestBuilder = originalRequest.newBuilder()
            .header("apikey", apikey)
            .header("Authorization", "Bearer $apikey") // Adding Authorization as per standard Supabase practice mentioned in theory
            .header("Prefer", "return=representation")
        val request = requestBuilder.build()
        return chain.proceed(request)
    }
}
