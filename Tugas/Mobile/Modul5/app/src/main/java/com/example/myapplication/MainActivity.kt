package com.example.myapplication

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Scaffold
import androidx.compose.ui.Modifier
import com.example.myapplication.ui.theme.MyApplicationTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        val gadgets = listOf(
            Gadget(1, "Smartphone", "High-end smartphone with great camera"),
            Gadget(2, "Laptop", "Powerful laptop for gaming and work"),
            Gadget(3, "Tablet", "Slim tablet with a beautiful display"),
            Gadget(4, "Smartwatch", "Keep track of your health and notifications"),
            Gadget(5, "Headphones", "Noise-cancelling over-ear headphones"),
            Gadget(6, "Camera", "Mirrorless camera for professional photography"),
            Gadget(7, "Speaker", "Portable Bluetooth speaker with deep bass"),
            Gadget(8, "Monitor", "4K Ultra HD Monitor"),
            Gadget(9, "Keyboard", "Mechanical wireless keyboard"),
            Gadget(10, "Mouse", "Ergonomic gaming mouse")
        )
        setContent {
            MyApplicationTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->

                    /* GadgetGridFixed(
                        gadgets = gadgets,
                        modifier = Modifier.padding(innerPadding)
                    ) */


                    GadgetGrid(
                    gadgets = gadgets,
                    modifier = Modifier.padding(innerPadding)
                    )
                }
            }
        }
    }
}
