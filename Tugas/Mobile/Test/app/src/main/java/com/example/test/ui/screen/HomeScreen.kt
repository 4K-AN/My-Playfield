package com.example.test.ui.screen

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

@Composable
fun HomeScreen(onOpenDetail: (Int) -> Unit) {
    val items = (1..20).toList()

    LazyColumn {
        items(items) { id ->
            Text(
                text = "Item #$id",
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable { onOpenDetail(id) }
                    .padding(16.dp)
            )
        }
    }
}
