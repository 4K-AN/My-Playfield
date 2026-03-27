package com.example.myapplication

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil.compose.AsyncImage

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            // Nilai diubah dari 0 menjadi 5 agar terhindar dari exception pembagian nol
            val angka = 5

            Column(
                modifier = Modifier.fillMaxSize(),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.Center
            ) {
                Text(text = "Tugas Praktikum Compose", fontSize = 24.sp)
                Spacer(modifier = Modifier.height(16.dp))

                // Menampilkan gambar dari internet menggunakan library Coil
                AsyncImage(
                    model = "https://picsum.photos/800/600",
                    contentDescription = null
                )

                Spacer(modifier = Modifier.height(16.dp))
                val hasil = 100 / angka
                Text(text = "Hasil: $hasil")
            }
        }
    }
}