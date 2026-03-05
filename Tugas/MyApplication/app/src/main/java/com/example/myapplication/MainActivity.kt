package com.example.myapplication

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.myapplication.ui.theme.MyApplicationTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            MyApplicationTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    CounterContent(modifier = Modifier.padding(innerPadding))
                }
            }
        }
    }
}

@Composable
fun CounterContent(modifier: Modifier = Modifier) {
    var counter by remember { mutableStateOf(0) }
    Column(
        modifier = modifier.fillMaxSize(),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(text = "$counter", fontSize = 172.sp)
        Button(onClick = { counter++ }) {
            Text("Add")
        }
        Button(onClick = { counter = 0 }) {
            Text("Reset")
        }
    }
}

@Composable
fun HelloContent(modifier: Modifier = Modifier) {
    var name by remember { mutableStateOf("") }
    Column(modifier = modifier.padding(horizontal = 16.dp)) {
        Text(
            text = "Hello, $name!",
            modifier = Modifier.padding(bottom = 8.dp),
            style = MaterialTheme.typography.bodyMedium
        )
        OutlinedTextField(
            value = name,
            onValueChange = { name = it },
            label = { Text("Name") }
        )
    }
}

@Preview(showBackground = true)
@Composable
fun CounterPreview() {
    MyApplicationTheme {
        CounterContent()
    }
}

@Preview(showBackground = true)
@Composable
fun HelloContentPreview() {
    MyApplicationTheme {
        HelloContent()
    }
}
