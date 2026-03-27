package com.example.stateandeventhandling

import android.util.Log
import androidx.compose.foundation.layout.Column
import androidx.compose.material3.Button
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue

@Composable
fun FormScreen() {
    var name by remember { mutableStateOf("") }
    
    Column {
        OutlinedTextField(
            value = name,
            onValueChange = {
                name = it
                Log.d("INPUT", "Nama : $name")
            },
            label = { Text("Nama") }
        )
        
        Button(onClick = {
            Log.d("BUTTON", "Button ditekan oleh $name")
        }) {
            Text("Submit")
        }
    }
}
