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
import androidx.compose.runtime.saveable.Saver
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.myapplication.ui.theme.MyApplicationTheme

data class User(val name: String, val age: Int)

val userSaver = Saver<User, Map<String, Any>>(
    save = { mapOf("name" to it.name, "age" to it.age) },
    restore = { User(it["name"] as String, it["age"] as Int) }
)

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            MyApplicationTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    HelloScreen(modifier = Modifier.padding(innerPadding))
                }
            }
        }
    }
}

@Composable
fun HelloScreen(modifier: Modifier = Modifier) {
    var name by rememberSaveable { mutableStateOf("") }
    HelloContent(
        name = name,
        onNameChange = { name = it },
        modifier = modifier
    )
}

@Composable
fun HelloContent(
    name: String,
    onNameChange: (String) -> Unit,
    modifier: Modifier = Modifier
) {
    Column(modifier = modifier.padding(16.dp)) {
        Text(
            text = "Hello, $name",
            style = MaterialTheme.typography.bodyMedium
        )
        OutlinedTextField(
            value = name,
            onValueChange = onNameChange,
            label = { Text("Name") }
        )
    }
}

@Composable
fun CustomSaverExample(modifier: Modifier = Modifier) {
    var user by rememberSaveable(stateSaver = userSaver) {
        mutableStateOf(User(name = "Budi", age = 28))
    }

    Column(
        modifier = modifier.padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Text(text = "Name: ${user.name}, Age: ${user.age}")
        Button(
            onClick = { user = user.copy(age = user.age + 1) }
        ) {
            Text("Increase Age")
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

@Preview(showBackground = true)
@Composable
fun HelloScreenPreview() {
    MyApplicationTheme {
        HelloScreen()
    }
}

@Preview(showBackground = true)
@Composable
fun CustomSaverPreview() {
    MyApplicationTheme {
        CustomSaverExample()
    }
}

@Preview(showBackground = true)
@Composable
fun CounterPreview() {
    MyApplicationTheme {
        CounterContent()
    }
}
