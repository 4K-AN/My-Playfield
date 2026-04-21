package com.example.myapplication.view

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.navigation.NavController

@Composable
fun ProfileScreen(navController: NavController, onEditClick: () -> Unit) {

    val newName = navController.currentBackStackEntry
        ?.savedStateHandle
        ?.getStateFlow<String?>("new_name", null)
        ?.collectAsState()

    var username by remember { mutableStateOf("User Default") }


    LaunchedEffect(newName?.value) {
        newName?.value?.let {
            username = it
        }
    }

    Column(
        modifier = Modifier.fillMaxSize().padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text(text = "Profile Screen", style = MaterialTheme.typography.headlineMedium)
        Spacer(modifier = Modifier.height(16.dp))
        Text(text = "Username: $username", style = MaterialTheme.typography.bodyLarge)
        Spacer(modifier = Modifier.height(24.dp))
        Button(onClick = onEditClick) {
            Text("Edit Username")
        }
    }
}
