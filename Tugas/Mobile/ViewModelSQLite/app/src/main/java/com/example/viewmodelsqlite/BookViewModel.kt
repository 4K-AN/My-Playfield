package com.example.viewmodelsqlite

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.launch

class BookViewModel(private val repository: BookRepository) : ViewModel() {
    fun getAllBook() = repository.getAllBook()

    fun insert(book: Book) {
        viewModelScope.launch {
            repository.insertBook(book)
        }
    }
}
