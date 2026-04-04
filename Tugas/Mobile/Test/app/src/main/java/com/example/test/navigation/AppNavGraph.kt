package com.example.test.navigation

import androidx.compose.runtime.Composable
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import androidx.navigation.navArgument
import com.example.test.DetailsScreen
import com.example.test.HomeScreen

@Composable
fun NavigationDemoApp() {
    val navController = rememberNavController()

    NavHost(
        navController = navController,
        startDestination = Routes.home
    ) {
        composable(Routes.home) {
            HomeScreen(
                onOpenDetail = { id -> 
                    navController.navigate("${Routes.detailBase}/$id") 
                }
            )
        }
        
        composable(
            route = Routes.detail,
            arguments = listOf(navArgument("id") { type = NavType.IntType })
        ) { backStackEntry ->
            val id = backStackEntry.arguments?.getInt("id") ?: 0
            DetailsScreen(
                itemId = id,
                onBack = { navController.popBackStack() }

                
                
            )
        }
    }
}
