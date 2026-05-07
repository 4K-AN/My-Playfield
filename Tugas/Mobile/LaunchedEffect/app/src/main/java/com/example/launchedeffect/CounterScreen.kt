package com.example.launchedeffect

import androidx.compose.foundation.layout.Column
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel

@Composable
fun CounterScreen(viewModel: CounterViewModel = viewModel()) {

    val angka by viewModel.angka.collectAsStateWithLifecycle()

    Column {
        Text(text = "Angka: $angka")
        Button(onClick = { viewModel.tambahAngka() }) {

            Text("tambah")
        }
    }
}
