package com.example.fooddiary.ui.navigation

object NavRoutes {
    const val SPLASH = "splash"
    const val AUTH = "auth"
    const val HOME = "home"
    const val DETAIL = "detail/{itemId}"
    const val FORM = "form?itemId={itemId}"
    const val PROFILE = "profile"

    fun detailRoute(itemId: String) = "detail/$itemId"
    fun formRoute(itemId: String? = null) =
        if (itemId != null) "form?itemId=$itemId" else "form"
}
