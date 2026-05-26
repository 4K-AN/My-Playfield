package com.example.fooddiary.model

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
data class FoodItem(
    val id: String = "",
    @SerialName("user_id") val userId: String = "",
    @SerialName("category_id") val categoryId: String? = null,
    val title: String = "",
    val description: String = "",
    @SerialName("image_url") val imageUrl: String? = null,
    @SerialName("created_at") val createdAt: String? = null
)
