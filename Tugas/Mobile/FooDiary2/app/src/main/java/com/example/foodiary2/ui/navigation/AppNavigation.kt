package com.example.foodiary2.ui.navigation

import androidx.compose.runtime.Composable
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import androidx.navigation.navArgument
import com.example.foodiary2.ui.screens.AuthScreen
import com.example.foodiary2.ui.screens.DetailScreen
import com.example.foodiary2.ui.screens.FormScreen
import com.example.foodiary2.ui.screens.HomeScreen
import com.example.foodiary2.ui.screens.ProfileScreen
import com.example.foodiary2.ui.screens.SplashScreen

@Composable
fun AppNavigation() {
    val navController = rememberNavController()

    NavHost(
        navController = navController,
        startDestination = NavRoutes.SPLASH
    ) {
        composable(NavRoutes.SPLASH) {
            SplashScreen(
                onNavigateToAuth = {
                    navController.navigate(NavRoutes.AUTH) {
                        popUpTo(NavRoutes.SPLASH) { inclusive = true }
                    }
                },
                onNavigateToHome = {
                    navController.navigate(NavRoutes.HOME) {
                        popUpTo(NavRoutes.SPLASH) { inclusive = true }
                    }
                }
            )
        }

        composable(NavRoutes.AUTH) {
            AuthScreen(
                onAuthSuccess = {
                    navController.navigate(NavRoutes.HOME) {
                        popUpTo(NavRoutes.AUTH) { inclusive = true }
                    }
                }
            )
        }

        composable(NavRoutes.HOME) {
            HomeScreen(
                onItemClick = { itemId ->
                    navController.navigate(NavRoutes.detailRoute(itemId))
                },
                onAddClick = {
                    navController.navigate(NavRoutes.formRoute())
                },
                onProfileClick = {
                    navController.navigate(NavRoutes.PROFILE)
                }
            )
        }

        composable(
            route = NavRoutes.DETAIL,
            arguments = listOf(navArgument("itemId") { type = NavType.StringType })
        ) { backStackEntry ->
            val itemId = backStackEntry.arguments?.getString("itemId") ?: return@composable
            DetailScreen(
                itemId = itemId,
                onEditClick = {
                    navController.navigate(NavRoutes.formRoute(itemId))
                },
                onDeleteSuccess = {
                    navController.popBackStack()
                },
                onBackClick = {
                    navController.popBackStack()
                }
            )
        }

        composable(
            route = NavRoutes.FORM,
            arguments = listOf(
                navArgument("itemId") {
                    type = NavType.StringType
                    nullable = true
                    defaultValue = null
                }
            )
        ) { backStackEntry ->
            val itemId = backStackEntry.arguments?.getString("itemId")
            FormScreen(
                itemId = itemId,
                onSaveSuccess = {
                    navController.popBackStack()
                },
                onBackClick = {
                    navController.popBackStack()
                }
            )
        }

        composable(NavRoutes.PROFILE) {
            ProfileScreen(
                onLogout = {
                    navController.navigate(NavRoutes.AUTH) {
                        popUpTo(0) { inclusive = true }
                    }
                },
                onBackClick = {
                    navController.popBackStack()
                }
            )
        }
    }
}
