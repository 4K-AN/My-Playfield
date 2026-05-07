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
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
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
                        LaunchedEffectKey()
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
fun LaunchedEffectKey() {
    var angka by remember { mutableIntStateOf(0) }
    var keterangan by remember { mutableStateOf("Tekan tombol") }

    LaunchedEffect(angka) {
        if (angka > 0) {
            keterangan = "Proses angka $angka..."
            delay(1000)
            keterangan = "Angka $angka selesai!"
        }
    }

    Column {
        Text(text = keterangan)
        Button(onClick = { angka++ }) {
            Text("Tambah Angka")
        }
    }
}

@Preview(showBackground = true)
@Composable
fun LaunchedEffectUnitPreview() {
    LaunchedEffectTheme {
        Column {
            LaunchedEffectUnit()
            LaunchedEffectKey()
        }
    }
}
