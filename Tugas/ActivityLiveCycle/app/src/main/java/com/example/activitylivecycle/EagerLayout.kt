package com.example.activitylivecycle

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Card
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

@Composable
fun EagerLayoutExample() {
    // Membuat 50 data otomatis
    val gadgetList = List(50) { index -> 
        Gadget("Smartphone Seri $index", "Rp ${10 + index}.000.000") 
    }
    
    val scrollState = rememberScrollState()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(scrollState)
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        // Render semua data sekaligus (Eager)
        gadgetList.forEach { gadget ->
            Card(modifier = Modifier.fillMaxWidth()) {
                Text(
                    text = "${gadget.merk} - ${gadget.harga}", 
                    modifier = Modifier.padding(16.dp)
                )
            }
        }
    }
}