package com.example.stateandeventhandling

import android.util.Log
import androidx.compose.foundation.layout.Column
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue

// Percobaan 1: Variabel Biasa
@Composable
fun CounterTest() {
    var count = 0
    Column {
        Text("Counter : $count")
        Button(onClick = {
            count++
            Log.d("COUNTER", "count = $count")
        }) {
            Text("Tambah")
        }
    }
}

// Percobaan 2: Menggunakan State
@Composable
fun CounterStateTest() {
    var count by remember { mutableStateOf(0) }
    Column {
        Text("Counter : $count")
        Button(onClick = {
            count++
            Log.d("COUNTER", "count = $count")
        }) {
            Text("Tambah")
        }
    }
}
