package com.example.foodiary2.model

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
data class Category(
    val id: String,
    val name: String,
    @SerialName("created_at") val createdAt: String? = null
)
