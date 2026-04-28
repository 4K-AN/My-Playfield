package com.example.myapplication.view

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.example.myapplication.viewmodel.User

@Composable
fun UserPanel(user: User) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 16.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(text = "Membership", style = MaterialTheme.typography.titleLarge)
            HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(text = "User Name:", style = MaterialTheme.typography.bodyMedium)
                Text(text = user.name, fontWeight = androidx.compose.ui.text.font.FontWeight.Bold)
            }
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(text = "Status:", style = MaterialTheme.typography.bodyMedium)
                Text(text = user.status, color = MaterialTheme.colorScheme.secondary)
            }
        }
    }
}

@Composable
fun CalculatorScreen(
    modifier: Modifier = Modifier,
    discount: Int = 0,
    user: User,
    onClick: (Int) -> Unit,
    onReset: () -> Unit
) {
    var number by rememberSaveable { mutableStateOf("") }

    Column(
        modifier = modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text(text = "Discount Calculator", style = MaterialTheme.typography.headlineMedium)
        
        Spacer(modifier = Modifier.height(24.dp))
        
        OutlinedTextField(
            value = number,
            onValueChange = { number = it },
            label = { Text("Enter Amount") },
            modifier = Modifier.fillMaxWidth()
        )
        
        Spacer(modifier = Modifier.height(16.dp))
        
        Text(
            text = "Discount: $discount",
            style = MaterialTheme.typography.headlineSmall,
            color = MaterialTheme.colorScheme.primary
        )
        
        Spacer(modifier = Modifier.height(24.dp))

        Button(
            onClick = { 
                val amount = number.toIntOrNull() ?: 0
                onClick(amount) 
            },
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("Calculate")
        }

        Spacer(modifier = Modifier.height(8.dp))

        Button(
            onClick = { 
                number = "0"
                onReset()
            },
            colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.secondary),
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("Reset")
        }

        // Penambahan UserPanel di bagian bawah
        UserPanel(user = user)
    }
}
