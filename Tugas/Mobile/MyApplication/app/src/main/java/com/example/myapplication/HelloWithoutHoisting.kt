package com.example.myapplication

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.example.myapplication.ui.theme.MyApplicationTheme


@Composable
fun HelloScreenWithoutHoisting(modifier: Modifier = Modifier) {
    HelloContentWithoutHoisting(modifier)
}

@Composable
fun HelloContentWithoutHoisting(modifier: Modifier = Modifier) {
    var name by rememberSaveable { mutableStateOf("") }
    
    Column(modifier = modifier.padding(16.dp)) {
        Text(text = "Hello, $name")
        OutlinedTextField(
            value = name,
            onValueChange = { name = it },
            label = { Text("Name") }
        )
    }
}

@Preview(showBackground = true)
@Composable
fun WithoutHoistingPreview() {
    MyApplicationTheme {
        HelloScreenWithoutHoisting()
    }
}
