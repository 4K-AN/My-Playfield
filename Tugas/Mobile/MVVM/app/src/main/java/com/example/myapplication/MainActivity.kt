package com.example.myapplication

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import com.example.myapplication.ui.theme.MyApplicationTheme

class MainActivity : ComponentActivity() {
    private val viewModel: CalculatorViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            MyApplicationTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    CalculatorScreen(
                        modifier = Modifier.padding(innerPadding),
                        viewModel = viewModel
                    )
                }
            }
        }
    }
}

@Composable
fun CalculatorScreen(
    modifier: Modifier = Modifier,
    viewModel: CalculatorViewModel
) {
    // State lokal di View
    var number by remember { mutableStateOf("0") }
    
    // State dari ViewModel (Single Source of Truth yang belum terikat sempurna di Reset)
    val discount by viewModel.discount.collectAsState()

    Column(
        modifier = modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        TextField(
            value = number,
            onValueChange = { number = it },
            label = { Text("Enter Amount") },
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
            modifier = Modifier.fillMaxWidth()
        )
        
        Spacer(Modifier.size(16.dp))
        
        Text(text = "Discount: $discount", style = MaterialTheme.typography.headlineMedium)
        
        Spacer(Modifier.size(16.dp))
        
        Row {
            Button(onClick = { 
                val amount = number.toDoubleOrNull() ?: 0.0
                viewModel.calculateDiscount(amount) 
            }) {
                Text(text = "Calculate")
            }
            
            Spacer(Modifier.size(8.dp))
            
            Button(onClick = { 
                number = "0" // Hanya mengubah state lokal di View
            }) { 
                Text(text = "Reset")
            }
        }
    }
}
