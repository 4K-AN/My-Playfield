package com.example.myapplication.navigation

import android.os.Parcelable
import kotlinx.parcelize.Parcelize
import kotlinx.serialization.Serializable

@Serializable
object Home

@Serializable
data class Details(
    val locationId: Int,
    val locationName: String,
    val isPremium: Boolean = false
)

@Serializable
object Profile

@Serializable
object EditUsername

@Serializable
object Calculator

@Parcelize
@Serializable
data class UserConfig(
    val username: String,
    val birthDate: String
) : Parcelable
