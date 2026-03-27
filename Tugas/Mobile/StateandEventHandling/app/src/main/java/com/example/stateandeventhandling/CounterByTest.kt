package com.example.stateandeventhandling

import androidx.compose.foundation.layout.Column
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue

// Percobaan 1: Tanpa by
@Composable
fun CounterWithoutBy() {
    val count = remember { mutableStateOf(0) }
    Column {
        Text("Counter : ${count.value}")
        Button(onClick = {
            count.value++
        }) {
            Text("Tambah")
        }
    }
}

// Percobaan 2: Menggunakan by
@Composable
fun CounterWithBy() {
    var count by remember { mutableStateOf(0) }
    Column {
        Text("Counter : $count")
        Button(onClick = {
            count++
        }) {
            Text("Tambah")
        }
    }
}
