package com.example.foodiary2.model

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
data class Profile(
    val id: String,
    @SerialName("full_name") val fullName: String = "",
    @SerialName("avatar_url") val avatarUrl: String? = null,
    @SerialName("updated_at") val updatedAt: String? = null
)
