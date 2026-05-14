package com.example.viewmodelsqlite

import kotlinx.coroutines.flow.Flow

class BookRepository(private val bookDatabase: BookDatabase) {
    fun getAllBook(): Flow<List<Book>> = bookDatabase.getBookDao().getAllBooks()
    
    suspend fun insertBook(book: Book) =
        bookDatabase.getBookDao().insertBook(book)
}
