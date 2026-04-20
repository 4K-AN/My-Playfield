package com.example.myapplication.view

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp

@Composable
fun DetailsScreen(locationId: Int, locationName: String, isPremium: Boolean, onBack: () -> Unit) {

    val backgroundColor = if (isPremium) Color(0xFFFFD700) else MaterialTheme.colorScheme.background

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(backgroundColor)
            .padding(16.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(text = "Destination Details", style = MaterialTheme.typography.headlineSmall)
        Spacer(modifier = Modifier.height(8.dp))
        Text(text = "Name: $locationName", fontWeight = FontWeight.Bold)
        Text(text = "ID: $locationId")
        Text(text = "Premium Access: $isPremium")
        Spacer(modifier = Modifier.height(24.dp))
        Button(onClick = onBack) { Text("Go Back") }
    }
}
