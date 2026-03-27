package com.example.activitylivecycle

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Card
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

@Composable
fun LazyColumnExample() {
    val gadgetList = List(50) { index -> 
        Gadget("Laptop Seri $index", "Rp ${15 + index}.000.000") 
    }

    LazyColumn(
        modifier = Modifier.fillMaxSize(),
        contentPadding = PaddingValues(16.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        items(gadgetList) { gadget ->
            Card(modifier = Modifier.fillMaxWidth()) {
                Text(
                    text = "${gadget.merk} - ${gadget.harga}", 
                    modifier = Modifier.padding(16.dp)
                )
            }
        }
    }
}