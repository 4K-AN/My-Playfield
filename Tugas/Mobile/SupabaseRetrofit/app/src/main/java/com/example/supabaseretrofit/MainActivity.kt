package com.example.supabaseretrofit

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.example.supabaseretrofit.ui.theme.SupabaseRetrofitTheme

class MainActivity : ComponentActivity() {
    private lateinit var bookViewModel: BookViewModel
    private lateinit var repository: BookRepository
    private lateinit var factory: BookViewModelFactory

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        repository = BookRepository()
        factory = BookViewModelFactory(repository)
        bookViewModel = ViewModelProvider(this, factory)[BookViewModel::class.java]
        
        enableEdgeToEdge()
        setContent {
            SupabaseRetrofitTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    BookScreen(innerPadding, bookViewModel)
                }
            }
        }
    }
}

@Composable
fun BookScreen(padding: PaddingValues, viewModel: BookViewModel) {
    val bookUiState: UIState<List<Book>> by viewModel.bookUiState.collectAsStateWithLifecycle()
    var _title by remember { mutableStateOf("") }
    var _isbn by remember { mutableStateOf("") }

    LaunchedEffect(Unit) {
        viewModel.listBooks()
    }

    Column(
        modifier = Modifier
            .padding(padding)
            .fillMaxWidth()
    ) {
        TextField(
            modifier = Modifier.fillMaxWidth(),
            value = _title,
            onValueChange = { _title = it },
            label = { Text("JUDUL") },
        )
        TextField(
            modifier = Modifier.fillMaxWidth(),
            value = _isbn,
            onValueChange = { _isbn = it },
            label = { Text("ISBN") },
        )
        Button(onClick = {
            val book = Book(
                judul = _title, isbn = _isbn,
                id = null
            )
            viewModel.insertBook(book)
        }) { Text(text = "SIMPAN") }

        when (bookUiState) {
            is UIState.Loading -> ShowLoading()
            is UIState.Success -> ShowList((bookUiState as UIState.Success<List<Book>>).data)
            is UIState.Error -> ShowError(text = (bookUiState as UIState.Error).message)
        }
    }
}

@Composable
fun ShowList(list: List<Book>) {
    LazyColumn(modifier = Modifier.fillMaxSize()) {
        items(list) { item ->
            Card(
                modifier = Modifier
                    .padding(vertical = 5.dp)
                    .fillMaxWidth()
            ) {
                Column(modifier = Modifier.padding(horizontal = 10.dp)) {
                    Text(text = item.judul)
                    Text(text = item.isbn)
                }
            }
        }
    }
}

@Composable
fun ShowLoading() {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .fillMaxHeight()
    ) {
        val contentDesc = "LOADING"
        CircularProgressIndicator(
            modifier = Modifier
                .align(Alignment.Center)
                .semantics {
                    contentDescription = contentDesc
                })
    }
}

@Composable
fun ShowError(text: String) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .fillMaxHeight(),
        verticalArrangement = Arrangement.spacedBy(10.dp, Alignment.CenterVertically),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = text,
            style = MaterialTheme.typography.titleMedium,
            color = Color.Gray,
            fontSize = 16.sp,
            modifier = Modifier.padding(4.dp)
        )
    }
}
