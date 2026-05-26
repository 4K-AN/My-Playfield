package com.example.fooddiary.data

import android.content.ContentResolver
import android.net.Uri
import com.example.fooddiary.SupabaseClient
import com.example.fooddiary.model.FoodItem
import com.example.fooddiary.model.Profile
import io.github.jan_tennert.supabase.auth.provider.builtin.Email
import java.util.UUID

object FoodRepository {

    private val client get() = SupabaseClient.client

    suspend fun signIn(email: String, password: String) {
        client.auth.signInWith(Email) {
            this.email = email
            this.password = password
        }
    }

    suspend fun signUp(email: String, password: String) {
        client.auth.signUpWith(Email) {
            this.email = email
            this.password = password
        }
    }

    suspend fun signOut() {
        client.auth.signOut()
    }

    suspend fun getCurrentUserId(): String? {
        return try {
            client.auth.currentUserOrNull()?.id
        } catch (_: Exception) { null }
    }

    suspend fun getCurrentUserEmail(): String? {
        return try {
            client.auth.currentUserOrNull()?.email
        } catch (_: Exception) { null }
    }

    suspend fun getProfile(userId: String): Profile? {
        return try {
            client.postgrest["profiles"].select {
                eq("id", userId)
            }.decodeSingle<Profile>()
        } catch (_: Exception) { null }
    }

    suspend fun upsertProfile(profile: Profile) {
        client.postgrest["profiles"].upsert(profile)
    }

    suspend fun getFoodItems(userId: String): List<FoodItem> {
        return try {
            client.postgrest["food_items"].select {
                eq("user_id", userId)
                order("created_at", ascending = false)
            }.decodeList<FoodItem>()
        } catch (_: Exception) { emptyList() }
    }

    suspend fun getFoodItem(id: String): FoodItem? {
        return try {
            client.postgrest["food_items"].select {
                eq("id", id)
            }.decodeSingle<FoodItem>()
        } catch (_: Exception) { null }
    }

    suspend fun upsertFoodItem(item: FoodItem) {
        client.postgrest["food_items"].upsert(item)
    }

    suspend fun deleteFoodItem(id: String) {
        client.postgrest["food_items"].delete {
            eq("id", id)
        }
    }

    suspend fun uploadImage(userId: String, imageBytes: ByteArray, fileName: String): String {
        val path = "$userId/$fileName"
        client.storage["food-images"].upload(path, imageBytes, upsert = true)
        return client.storage["food-images"].publicUrl(path)
    }

    fun generateFileName(): String = "${UUID.randomUUID()}.jpg"

    fun readImageBytes(contentResolver: ContentResolver, uri: Uri): ByteArray? {
        return try {
            contentResolver.openInputStream(uri)?.use { it.readBytes() }
        } catch (_: Exception) { null }
    }
}
