package com.example.lazylayout

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.example.lazylayout.ui.theme.LazyLayoutTheme

@Composable
fun LazyGridExample() {
    LazyVerticalGrid(
        columns = GridCells.Adaptive(minSize = 150.dp), // Menyesuaikan kolom otomatis
        modifier = Modifier.fillMaxSize(),
        contentPadding = PaddingValues(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
        horizontalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        items(menuMakanan) { menu ->
            Column(modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp)) {
                Text(text = menu.nama, fontWeight = FontWeight.Bold)
                Text(text = menu.harga)
            }
        }
    }
}

@Preview(showBackground = true)
@Composable
fun LazyGridExamplePreview() {
    LazyLayoutTheme {
        LazyGridExample()
    }
}
