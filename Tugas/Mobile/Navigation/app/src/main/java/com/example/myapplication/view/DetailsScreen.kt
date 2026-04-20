package com.example.myapplication.view

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp

@Composable
fun DetailsScreen(locationId: Int, locationName: String, isPremium: Boolean) {
    Column(
        modifier = Modifier.fillMaxSize().padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text(text = "Detail Lokasi", style = MaterialTheme.typography.headlineMedium)
        Spacer(modifier = Modifier.height(8.dp))
        Text(text = "ID: $locationId", style = MaterialTheme.typography.bodyLarge)
        Text(text = "Nama: $locationName", style = MaterialTheme.typography.bodyLarge)
        
        Spacer(modifier = Modifier.height(16.dp))
        
        if (isPremium) {
            Surface(
                color = Color(0xFFFFD700), // Gold
                shape = MaterialTheme.shapes.medium
            ) {
                Text(
                    text = "PREMIUM ACCESS ENABLED",
                    modifier = Modifier.padding(8.dp),
                    color = Color.Black,
                    style = MaterialTheme.typography.labelLarge
                )
            }
        } else {
            Text(text = "Standard Access", color = Color.Gray)
        }
    }
}
