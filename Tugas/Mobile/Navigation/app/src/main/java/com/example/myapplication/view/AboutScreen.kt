package com.example.myapplication.view

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp

@Composable
fun AboutScreen() {
    Column(
        modifier = Modifier.fillMaxSize().padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text(text = "Tentang Saya", style = MaterialTheme.typography.headlineMedium)
        Spacer(modifier = Modifier.height(16.dp))
        Text(text = "Nama: Akhmad Syafiul Anam", style = MaterialTheme.typography.bodyLarge)
        Text(text = "NIM: 245150707111012", style = MaterialTheme.typography.bodyLarge)
    }
}

@Composable
fun AboutScreenDynamic(name: String, nim: String) {
    Column(
        modifier = Modifier.fillMaxSize().padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text(text = "Profil Mahasiswa", style = MaterialTheme.typography.headlineMedium)
        Spacer(modifier = Modifier.height(16.dp))
        Text(text = "Nama: $name", fontWeight = FontWeight.Bold, style = MaterialTheme.typography.bodyLarge)
        Text(text = "NIM: $nim", style = MaterialTheme.typography.bodyLarge)
    }
}
