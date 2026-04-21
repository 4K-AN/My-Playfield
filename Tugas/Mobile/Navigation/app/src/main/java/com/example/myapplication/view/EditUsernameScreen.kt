package com.example.myapplication.view

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.navigation.NavController

@Composable
fun EditUsernameScreen(navController: NavController) {
    var newName by remember { mutableStateOf("") }

    Column(
        modifier = Modifier.fillMaxSize().padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text(text = "Edit Username", style = MaterialTheme.typography.headlineMedium)
        
        Spacer(modifier = Modifier.height(16.dp))
        
        OutlinedTextField(
            value = newName,
            onValueChange = { newName = it },
            label = { Text("New Username") },
            modifier = Modifier.fillMaxWidth()
        )
        
        Spacer(modifier = Modifier.height(16.dp))
        
        Button(
            onClick = {
                // Mengirimkan data kembali ke ProfileScreen via SavedStateHandle
                navController.previousBackStackEntry
                    ?.savedStateHandle
                    ?.set("new_name", newName)
                navController.popBackStack()
            },
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("Save")
        }
        
        // Tombol Cancel sesuai Soal No. 5
        Button(
            onClick = { 
                // Hanya kembali tanpa menyimpan data
                navController.popBackStack() 
            },
            colors = ButtonDefaults.buttonColors(containerColor = Color.Red),
            modifier = Modifier.fillMaxWidth().padding(top = 8.dp)
        ) { 
            Text("Cancel") 
        }
    }
}
