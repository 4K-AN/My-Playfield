package com.example.myapplication.view

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.example.myapplication.navigation.UserConfig

@Composable
fun EditUsernameScreen(
    onSave: (String, String) -> Unit,
    onCancel: () -> Unit
) {
    var newName by remember { mutableStateOf("") }
    var newDob by remember { mutableStateOf("") }

    Column(
        modifier = Modifier.fillMaxSize().padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text(text = "Edit Profile", style = MaterialTheme.typography.headlineMedium)
        
        Spacer(modifier = Modifier.height(16.dp))
        
        OutlinedTextField(
            value = newName,
            onValueChange = { newName = it },
            label = { Text("New Username") },
            modifier = Modifier.fillMaxWidth()
        )

        Spacer(modifier = Modifier.height(8.dp))

        OutlinedTextField(
            value = newDob,
            onValueChange = { newDob = it },
            label = { Text("Birth Date (DD-MM-YYYY)") },
            modifier = Modifier.fillMaxWidth()
        )
        
        Spacer(modifier = Modifier.height(16.dp))
        
        Button(
            onClick = { onSave(newName, newDob) },
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("Save")
        }
        
        Button(
            onClick = onCancel,
            colors = ButtonDefaults.buttonColors(containerColor = Color.Red),
            modifier = Modifier.fillMaxWidth().padding(top = 8.dp)
        ) { 
            Text("Cancel") 
        }
    }
}
