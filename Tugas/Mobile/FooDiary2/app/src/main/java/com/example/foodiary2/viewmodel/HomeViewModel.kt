package com.example.foodiary2.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.foodiary2.data.FoodRepository
import com.example.foodiary2.model.FoodItem
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

data class HomeUiState(
    val foodItems: List<FoodItem> = emptyList(),
    val isLoading: Boolean = false,
    val error: String? = null
)

class HomeViewModel : ViewModel() {

    private val _uiState = MutableStateFlow(HomeUiState())
    val uiState: StateFlow<HomeUiState> = _uiState.asStateFlow()

    fun loadItems() {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isLoading = true, error = null)
            try {
                val userId = FoodRepository.getCurrentUserId()
                if (userId != null) {
                    val items = FoodRepository.getFoodItems(userId)
                    _uiState.value = HomeUiState(foodItems = items)
                } else {
                    _uiState.value = HomeUiState(error = "Sesi tidak ditemukan")
                }
            } catch (e: Exception) {
                _uiState.value = _uiState.value.copy(
                    isLoading = false,
                    error = e.message ?: "Gagal memuat data"
                )
            }
        }
    }
}
