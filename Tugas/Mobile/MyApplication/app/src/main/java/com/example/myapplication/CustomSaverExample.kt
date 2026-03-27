package com.example.myapplication

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.Saver
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.example.myapplication.ui.theme.MyApplicationTheme


data class User(val name: String, val age: Int)

val userSaver = Saver<User, Map<String, Any>>(
    save = { mapOf("name" to it.name, "age" to it.age) },
    restore = { User(it["name"] as String, it["age"] as Int) }
)

@Composable
fun CustomSaverExample(modifier: Modifier = Modifier) {
    var user by rememberSaveable(stateSaver = userSaver) {
        mutableStateOf(User(name = "Budi", age = 28))
    }

    Column(modifier = modifier.padding(16.dp)) {
        Text(text = "Name: ${user.name}, Age: ${user.age}")
        Button(
            onClick = { user = user.copy(age = user.age + 1) }
        ) {
            Text("Increase Age")
        }
    }
}

@Preview(showBackground = true)
@Composable
fun CustomSaverPreview() {
    MyApplicationTheme {
        CustomSaverExample()
    }
}
