package com.example.launchedeffect

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.launchedeffect.ui.theme.LaunchedEffectTheme
import kotlinx.coroutines.delay

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            LaunchedEffectTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    Column(modifier = Modifier.padding(innerPadding)) {
                        LaunchedEffectUnit()
                        LaunchedEffectWithViewModel()
                    }
                }
            }
        }
    }
}

@Composable
fun LaunchedEffectUnit(modifier: Modifier = Modifier) {
    var status by remember { mutableStateOf("Memulai...") }

    LaunchedEffect(Unit) {
        delay(2000)
        status = "selesai"
    }

    Text(
        text = status,
        modifier = modifier
    )
}

@Composable
fun LaunchedEffectWithViewModel(viewModel: CounterViewModel = viewModel()) {
    // Mengambil data angka dari ViewModel
    val angka by viewModel.angka.collectAsStateWithLifecycle()
    var keterangan by remember { mutableStateOf("Tekan tombol") }

    // LaunchedEffect memantau perubahan 'angka' yang berasal dari ViewModel
    LaunchedEffect(angka) {
        if (angka > 0) {
            keterangan = "Proses angka $angka dari ViewModel..."
            delay(1000)
            keterangan = "Angka $angka selesai!"
        }
    }

    Column {
        Text(text = keterangan)
        Text(text = "Nilai di ViewModel: $angka")
        Button(onClick = { viewModel.tambahAngka() }) {
            Text("Tambah Angka")
        }
    }
}

@Preview(showBackground = true)
@Composable
fun LaunchedEffectPreview() {
    LaunchedEffectTheme {
        Column {
            LaunchedEffectUnit()
            LaunchedEffectWithViewModel()
        }
    }
}
