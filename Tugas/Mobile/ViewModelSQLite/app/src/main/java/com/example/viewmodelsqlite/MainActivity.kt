package com.example.viewmodelsqlite

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.example.viewmodelsqlite.ui.theme.ViewModelSQLiteTheme

class MainActivity : ComponentActivity() {
    private lateinit var bookViewModel: BookViewModel
    private lateinit var bookDatabase: BookDatabase
    private lateinit var repository: BookRepository
    private lateinit var factory: BookViewModelFactory

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Inisialisasi Database, Repository, dan ViewModel
        bookDatabase = BookDatabase(this)
        repository = BookRepository(bookDatabase)
        factory = BookViewModelFactory(repository)
        bookViewModel = ViewModelProvider(this, factory)[BookViewModel::class.java]

        enableEdgeToEdge()
        setContent {
            ViewModelSQLiteTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    Greeting(
                        modifier = Modifier.padding(innerPadding),
                        bookViewModel = bookViewModel
                    )
                }
            }
        }
    }
}

@Composable
fun BookCard(book: Book) {
    Card(
        modifier = Modifier
            .padding(vertical = 5.dp)
            .fillMaxWidth()
    ) {
        Column(modifier = Modifier.padding(horizontal = 10.dp)) {
            Text(text = "JUDUL: " + book.title)
            Text(text = "ISBN: " + book.isbn)
        }
    }
}

@Composable
fun Greeting(modifier: Modifier = Modifier, bookViewModel: BookViewModel) {
    val itemList by bookViewModel.getAllBook().collectAsStateWithLifecycle(initialValue = listOf())
    var _title by remember { mutableStateOf("") }
    var _isbn by remember { mutableStateOf("") }

    Column(modifier = modifier.padding(all = 16.dp)) {
        TextField(
            value = _title,
            onValueChange = { _title = it },
            label = { Text("JUDUL") },
            modifier = Modifier.fillMaxWidth()
        )
        TextField(
            value = _isbn,
            onValueChange = { _isbn = it },
            label = { Text("ISBN") },
            modifier = Modifier.fillMaxWidth().padding(top = 8.dp)
        )
        Button(
            onClick = {
                if (_title.isNotBlank() && _isbn.isNotBlank()) {
                    val book = Book(id = null, title = _title, isbn = _isbn)
                    bookViewModel.insert(book)
                    _title = ""
                    _isbn = ""
                }
            },
            modifier = Modifier.padding(top = 16.dp)
        ) {
            Text(text = "SIMPAN")
        }
        
        LazyColumn(modifier = Modifier.padding(top = 16.dp)) {
            items(itemList) { book ->
                BookCard(book)
            }
        }
    }
}
