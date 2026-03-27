package com.example.stateandeventhandling

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.stateandeventhandling.ui.theme.StateandEventHandlingTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            StateandEventHandlingTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    Column(
                        modifier = Modifier
                            .padding(innerPadding)
                            .padding(16.dp)
                            .verticalScroll(rememberScrollState())
                    ) {
                        // --- SILAHKAN AKTIFKAN/NONAKTIFKAN DENGAN KOMENTAR (//) ---

                        // Soal No. 1: Variabel Biasa vs State

                        /* Text("Soal No. 1: Variabel Biasa vs State", fontSize = 20.sp, fontWeight = FontWeight.Bold)
                        Text("Percobaan 1: Variabel Biasa", modifier = Modifier.padding(top = 8.dp))
                        CounterTest()
                        Text("Percobaan 2: Menggunakan State", modifier = Modifier.padding(top = 8.dp))
                        CounterStateTest()
                        HorizontalDivider(modifier = Modifier.padding(vertical = 16.dp), color = Color.Gray) */


                        // Soal No. 2: Penggunaan by dan tanpa by

                        /* Text("Soal No. 2: Penggunaan by dan tanpa by", fontSize = 20.sp, fontWeight = FontWeight.Bold)
                        Text("Percobaan 1: Tanpa by", modifier = Modifier.padding(top = 8.dp))
                        CounterWithoutBy()
                        Text("Percobaan 2: Menggunakan by", modifier = Modifier.padding(top = 8.dp))
                        CounterWithBy()
                        HorizontalDivider(modifier = Modifier.padding(vertical = 16.dp), color = Color.Gray) */


                        // Soal No. 3: Event Handling pada Compose
                       Text("Soal No. 3: Event Handling pada Compose", fontSize = 20.sp, fontWeight = FontWeight.Bold)
                        Text("Percobaan: Form Screen", modifier = Modifier.padding(top = 8.dp))
                        FormScreen()
                        
                        // ---------------------------------------------------------
                    }
                }
            }
        }
    }
}
