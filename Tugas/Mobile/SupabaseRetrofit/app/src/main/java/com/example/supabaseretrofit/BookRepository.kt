package com.example.supabaseretrofit

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn

class BookRepository {
    private val bookService = RetrofitInstance.bookService

    fun listBooks(): Flow<List<Book>> = flow {
        val res = bookService.getBooks()
        if (res.isSuccessful) {
            emit(res.body()!!)
        } else {
            throw Exception(res.errorBody()?.string())
        }
    }.flowOn(Dispatchers.IO)

    fun insertBook(book: Book): Flow<List<Book>> = flow {
        val res = bookService.addBook(book)
        if (res.isSuccessful) {
            emit(res.body()!!)
        } else {
            throw Exception(res.errorBody()?.string())
        }
    }.flowOn(Dispatchers.IO)
}
