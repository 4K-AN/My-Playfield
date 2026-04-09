package com.example.myapplication

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

data class Gadget(
    val id: Int,
    val name: String,
    val description: String
)

@Composable
fun GadgetItem(gadget: Gadget) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(8.dp)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(text = gadget.name, style = MaterialTheme.typography.headlineMedium)
            Text(text = gadget.description, style = MaterialTheme.typography.bodyMedium)
        }
    }
}

@Composable
fun GadgetListColumn(gadgets: List<Gadget>, modifier: Modifier = Modifier) {
    LazyColumn(
        contentPadding = PaddingValues(bottom = 16.dp),
        modifier = modifier.fillMaxSize()
    ) {
        items(gadgets) { gadget ->
            GadgetItem(gadget)
        }
    }
}

@Composable
fun GadgetListRow(gadgets: List<Gadget>, modifier: Modifier = Modifier) {
    LazyRow(
        contentPadding = PaddingValues(end = 16.dp),
        modifier = modifier.fillMaxSize()
    ) {
        items(gadgets) { gadget ->
            GadgetItem(gadget)
        }
    }
}
