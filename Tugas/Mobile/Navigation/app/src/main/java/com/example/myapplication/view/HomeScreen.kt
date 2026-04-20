package com.example.myapplication.view

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

@Composable
fun HomeScreen(onNavigateToDetails: (Int, String, Boolean) -> Unit) {
    val locations = listOf("Paris", "Tokyo", "New York", "London")

    var isPremium by remember { mutableStateOf(false) } 

    Column(
        modifier = Modifier.fillMaxSize().padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(text = "Travel Journal", style = MaterialTheme.typography.headlineMedium)
        
        Spacer(modifier = Modifier.height(16.dp))


        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("Premium User")
            Spacer(modifier = Modifier.width(8.dp))
            Switch(
                checked = isPremium,
                onCheckedChange = { isPremium = it }
            )
        }
        
        Spacer(modifier = Modifier.height(16.dp))
        locations.forEachIndexed { index, name ->
            Button(

                onClick = { onNavigateToDetails(index, name, isPremium) }, 
                modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp)
            ) { Text(text = "Visit $name") }
        }
    }
}
