package com.example.foodiary2.data

import android.content.ContentResolver
import android.net.Uri
import com.example.foodiary2.SupabaseClient
import com.example.foodiary2.model.FoodItem
import com.example.foodiary2.model.Profile
import io.github.jan.supabase.auth.auth
import io.github.jan.supabase.auth.providers.builtin.Email
import io.github.jan.supabase.postgrest.from
import io.github.jan.supabase.storage.storage
import java.util.UUID

object FoodRepository {

    private val client get() = SupabaseClient.client

    suspend fun signIn(emailInput: String, passwordInput: String) {
        client.auth.signInWith(Email) {
            email = emailInput
            password = passwordInput
        }
    }

    suspend fun signUp(emailInput: String, passwordInput: String) {
        client.auth.signUpWith(Email) {
            email = emailInput
            password = passwordInput
        }
    }

    suspend fun signOut() {
        client.auth.signOut()
    }

    /**
     * Returns current user ID.
     * Tries the cached session first, falls back to refreshing.
     */
    suspend fun getCurrentUserId(): String? {
        return try {
            client.auth.currentUserOrNull()?.id
                ?: client.auth.retrieveUserForCurrentSession(updateSession = true).id
        } catch (_: Exception) { null }
    }

    suspend fun getCurrentUserEmail(): String? {
        return try {
            client.auth.currentUserOrNull()?.email
                ?: client.auth.retrieveUserForCurrentSession(updateSession = true).email
        } catch (_: Exception) { null }
    }

    /** true when there is an active, valid session. */
    suspend fun hasActiveSession(): Boolean {
        return getCurrentUserId() != null
    }

    suspend fun getProfile(userId: String): Profile? {
        return try {
            client.from("profiles").select {
                filter {
                    eq("id", userId)
                }
            }.decodeSingle<Profile>()
        } catch (_: Exception) { null }
    }

    suspend fun upsertProfile(profile: Profile) {
        client.from("profiles").upsert(profile)
    }

    suspend fun getFoodItems(userId: String): List<FoodItem> {
        return try {
            client.from("food_items").select {
                filter {
                    eq("user_id", userId)
                }
            }.decodeList<FoodItem>()
        } catch (_: Exception) { emptyList() }
    }

    suspend fun getFoodItem(id: String): FoodItem? {
        return try {
            client.from("food_items").select {
                filter {
                    eq("id", id)
                }
            }.decodeSingle<FoodItem>()
        } catch (_: Exception) { null }
    }

    suspend fun upsertFoodItem(item: FoodItem) {
        client.from("food_items").upsert(item)
    }

    suspend fun deleteFoodItem(id: String) {
        client.from("food_items").delete {
            filter {
                eq("id", id)
            }
        }
    }

    suspend fun uploadImage(userId: String, imageBytes: ByteArray, fileName: String): String {
        val path = "$userId/$fileName"
        val bucket = client.storage.from("food_images")
        bucket.upload(path, imageBytes) {
            upsert = true
        }
        return bucket.publicUrl(path)
    }

    fun generateFileName(): String = "${UUID.randomUUID()}.jpg"

    fun readImageBytes(contentResolver: ContentResolver, uri: Uri): ByteArray? {
        return try {
            contentResolver.openInputStream(uri)?.use { it.readBytes() }
        } catch (_: Exception) { null }
    }
}
