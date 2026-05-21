package com.example.supabaseretrofit

import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider

class BookViewModelFactory(private val repository: BookRepository) :
    ViewModelProvider.NewInstanceFactory() {
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        try {
            val constructor =
                modelClass.getDeclaredConstructor(BookRepository::class.java)
            return constructor.newInstance(repository)
        } catch (e: Exception) {
            Log.e("ERROR", e.message.toString())
        }
        return super.create(modelClass)
    }
}
