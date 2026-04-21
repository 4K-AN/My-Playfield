package com.example.myapplication.view

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.navigation.NavController
import com.example.myapplication.navigation.UserConfig

@Composable
fun ProfileScreen(navController: NavController, onEditClick: () -> Unit) {

    val userConfigFlow = navController.currentBackStackEntry
        ?.savedStateHandle
        ?.getStateFlow<UserConfig?>("user_config_data", null)
        ?.collectAsState()

    var username by remember { mutableStateOf("User Default") }
    var birthDate by remember { mutableStateOf("-") }


    LaunchedEffect(userConfigFlow?.value) {
        userConfigFlow?.value?.let { config ->
            username = config.username
            birthDate = config.birthDate
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
        Text(text = "Birth Date: $birthDate", style = MaterialTheme.typography.bodyLarge)
        Spacer(modifier = Modifier.height(24.dp))
        Button(onClick = onEditClick) {
            Text("Edit Profile")
        }
    }
}
