package com.example.viewmodelsqlite

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.CreationExtras
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

class BookViewModel(private val bookDao: BookDao) : ViewModel() {

    val allBooks: StateFlow<List<Book>> = bookDao.getAllBooks()
        .stateIn(
            scope = viewModelScope,
            started = SharingStarted.WhileSubscribed(5000),
            initialValue = emptyList()
        )

    fun addBook(title: String, isbn: String) {
        viewModelScope.launch {
            bookDao.insertBook(Book(title = title, isbn = isbn))
        }
    }

    fun deleteBook(book: Book) {
        viewModelScope.launch {
            bookDao.deleteBook(book)
        }
    }

    companion object {
        val Factory: ViewModelProvider.Factory = object : ViewModelProvider.Factory {
            @Suppress("UNCHECKED_CAST")
            override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
                val application = checkNotNull(extras[ViewModelProvider.AndroidViewModelFactory.APPLICATION_KEY])
                val database = AppDatabase.getDatabase(application)
                return BookViewModel(database.bookDao()) as T
            }
        }
    }
}
