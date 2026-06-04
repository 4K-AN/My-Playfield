package com.example.foodiary2.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.foodiary2.data.FoodRepository
import com.example.foodiary2.model.FoodItem
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

data class DetailUiState(
    val foodItem: FoodItem? = null,
    val loadingState: LoadingState = LoadingState.Idle,
    val error: String? = null,
    val isDeleted: Boolean = false
) {
    val isLoading: Boolean get() = loadingState.isLoading
}

class DetailViewModel : ViewModel() {

    private val _uiState = MutableStateFlow(DetailUiState())
    val uiState: StateFlow<DetailUiState> = _uiState.asStateFlow()

    private var currentItem: FoodItem? = null

    fun loadItem(itemId: String) {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(loadingState = LoadingState.Loading, error = null)
            try {
                val item = FoodRepository.getFoodItem(itemId)
                currentItem = item
                _uiState.value = DetailUiState(foodItem = item)
            } catch (e: Exception) {
                _uiState.value = _uiState.value.copy(
                    loadingState = LoadingState.Idle,
                    error = e.message ?: "Gagal memuat detail"
                )
            }
        }
    }

    fun deleteItem() {
        val item = currentItem ?: return
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(loadingState = LoadingState.Deleting, error = null)
            try {
                FoodRepository.deleteFoodItem(item)
                _uiState.value = _uiState.value.copy(loadingState = LoadingState.Idle, isDeleted = true)
            } catch (e: Exception) {
                _uiState.value = _uiState.value.copy(
                    loadingState = LoadingState.Idle,
                    error = e.message ?: "Gagal menghapus"
                )
            }
        }
    }
}
