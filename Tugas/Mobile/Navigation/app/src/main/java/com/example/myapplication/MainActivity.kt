package com.example.myapplication

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material3.Button
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import androidx.navigation.toRoute
import com.example.myapplication.navigation.Details
import com.example.myapplication.navigation.Home
import com.example.myapplication.ui.theme.MyApplicationTheme
import com.example.myapplication.view.AboutScreen
import com.example.myapplication.view.AboutScreenDynamic
import com.example.myapplication.view.DetailsScreen
import com.example.myapplication.view.HomeScreen

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            MyApplicationTheme {
                val navController = rememberNavController()
                Scaffold(
                    modifier = Modifier.fillMaxSize(),
                    bottomBar = {
                        Row(modifier = Modifier.padding(16.dp)) {
                            Button(onClick = { navController.navigate(Home) }) {
                                Text("Home")
                            }
                            Spacer(modifier = Modifier.width(8.dp))
                            Button(onClick = { navController.navigate("about_screen") }) {
                                Text("About")
                            }
                            Spacer(modifier = Modifier.width(8.dp))
                            Button(onClick = { 
                                navController.navigate("about_screen/Akhmad Syafiul Anam/245150707111012") 
                            }) {
                                Text("About (Dyn)")
                            }
                        }
                    }
                ) { innerPadding ->
                    NavHost(
                        navController = navController,
                        startDestination = Home,
                        modifier = Modifier.padding(innerPadding)
                    ) {
                        composable<Home> {
                            HomeScreen(onNavigateToDetails = { id, name, premiumStatus ->
                                navController.navigate(
                                    route = Details(locationId = id, locationName = name, isPremium = premiumStatus)
                                )
                            })
                        }
                        composable<Details> { backStackEntry ->
                            val args = backStackEntry.toRoute<Details>()
                            DetailsScreen(
                                locationId = args.locationId,
                                locationName = args.locationName,
                                isPremium = args.isPremium,
                                onBack = { navController.popBackStack() }
                            )
                        }
                        
                        // Soal No. 5a: Rute String Statis
                        composable("about_screen") {
                            AboutScreen()
                        }

                        // Soal No. 5b: Rute String Dinamis
                        composable(route = "about_screen/{studentName}/{studentNim}") { backStackEntry ->
                            val name = backStackEntry.arguments?.getString("studentName") ?: "Unknown"
                            val nim = backStackEntry.arguments?.getString("studentNim") ?: "Unknown"
                            AboutScreenDynamic(name, nim)
                        }
                    }
                }
            }
        }
    }
}
