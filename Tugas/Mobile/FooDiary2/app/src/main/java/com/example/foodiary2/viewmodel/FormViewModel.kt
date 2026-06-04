package com.example.foodiary2.viewmodel

import android.content.ContentResolver
import android.net.Uri
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.foodiary2.data.FoodRepository
import com.example.foodiary2.model.Category
import com.example.foodiary2.model.FoodItem
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.util.UUID

data class FormUiState(
    val title: String = "",
    val description: String = "",
    val imageUri: Uri? = null,
    val categories: List<Category> = emptyList(),
    val selectedCategoryId: String? = null,
    val loadingState: LoadingState = LoadingState.Idle,
    val isEditMode: Boolean = false,
    val error: String? = null,
    val isSuccess: Boolean = false
) {
    val isLoading: Boolean get() = loadingState.isLoading
}

class FormViewModel : ViewModel() {

    private val _uiState = MutableStateFlow(FormUiState())
    val uiState: StateFlow<FormUiState> = _uiState.asStateFlow()

    private var existingItemId: String? = null
    private var existingImageUrl: String? = null
    private var selectedImageBytes: ByteArray? = null

    fun loadCategories() {
        viewModelScope.launch {
            val categories = FoodRepository.getCategories()
            _uiState.value = _uiState.value.copy(categories = categories)
        }
    }

    fun loadItem(itemId: String) {
        existingItemId = itemId
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(loadingState = LoadingState.Loading)
            try {
                val categories = FoodRepository.getCategories()
                val item = FoodRepository.getFoodItem(itemId)
                if (item != null) {
                    existingImageUrl = item.imageUrl
                    _uiState.value = FormUiState(
                        title = item.title,
                        description = item.description,
                        categories = categories,
                        selectedCategoryId = item.categoryId,
                        isEditMode = true
                    )
                }
            } catch (e: Exception) {
                _uiState.value = _uiState.value.copy(loadingState = LoadingState.Idle, error = e.message)
            }
        }
    }

    fun onTitleChanged(title: String) {
        _uiState.value = _uiState.value.copy(title = title, error = null)
    }

    fun onDescriptionChanged(description: String) {
        _uiState.value = _uiState.value.copy(description = description, error = null)
    }

    fun onCategorySelected(categoryId: String?) {
        _uiState.value = _uiState.value.copy(selectedCategoryId = categoryId, error = null)
    }

    fun onImageSelected(uri: Uri, contentResolver: ContentResolver) {
        selectedImageBytes = FoodRepository.readImageBytes(contentResolver, uri)
        _uiState.value = _uiState.value.copy(imageUri = uri, error = null)
    }

    fun save() {
        val state = _uiState.value
        if (state.title.isBlank()) {
            _uiState.value = state.copy(error = "Judul harus diisi")
            return
        }

        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(loadingState = LoadingState.Saving, error = null)
            try {
                val userId = FoodRepository.getCurrentUserId()
                    ?: throw Exception("Sesi tidak ditemukan")

                var imageUrl = existingImageUrl

                if (selectedImageBytes != null) {
                    val fileName = FoodRepository.generateFileName()
                    imageUrl = FoodRepository.uploadImage(userId, selectedImageBytes!!, fileName)
                }

                val item = FoodItem(
                    id = existingItemId ?: UUID.randomUUID().toString(),
                    userId = userId,
                    categoryId = state.selectedCategoryId,
                    title = state.title,
                    description = state.description,
                    imageUrl = imageUrl
                )

                FoodRepository.upsertFoodItem(item)
                _uiState.value = _uiState.value.copy(loadingState = LoadingState.Idle, isSuccess = true)
            } catch (e: Exception) {
                _uiState.value = _uiState.value.copy(
                    loadingState = LoadingState.Idle,
                    error = e.message ?: "Gagal menyimpan"
                )
            }
        }
    }
}
