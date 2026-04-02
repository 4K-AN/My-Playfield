package com.example.test.navigation

import androidx.compose.runtime.Composable
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.example.test.DetailsScreen
import com.example.test.HomeScreen

object Routes {
    const val home = "home"
    const val details = "details"
}

@Composable
fun NavigationDemoApp() {
    val navController = rememberNavController()

    NavHost(
        navController = navController,
        startDestination = Routes.home
    ) {
        composable(Routes.home) {
            HomeScreen(
                onNavigateToDetails = { navController.navigate(Routes.details) }
            )
        }
        composable(Routes.details) {
            DetailsScreen(
                onNavigateBack = { navController.popBackStack() }
            )
        }
    }
}
