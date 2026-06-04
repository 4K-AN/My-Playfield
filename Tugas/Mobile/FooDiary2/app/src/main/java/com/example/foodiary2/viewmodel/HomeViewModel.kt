package com.example.foodiary2.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.foodiary2.data.FoodRepository
import com.example.foodiary2.model.Category
import com.example.foodiary2.model.FoodItem
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

data class HomeUiState(
    val foodItems: List<FoodItem> = emptyList(),
    val categories: List<Category> = emptyList(),
    val selectedCategoryId: String? = null,
    val loadingState: LoadingState = LoadingState.Idle,
    val error: String? = null,
    val searchQuery: String = ""
) {
    val isLoading: Boolean get() = loadingState.isLoading

    val filteredItems: List<FoodItem>
        get() {
            var result = foodItems
            // Filter by search query
            if (searchQuery.isNotBlank()) {
                result = result.filter { item ->
                    item.title.contains(searchQuery, ignoreCase = true) ||
                    item.description.contains(searchQuery, ignoreCase = true)
                }
            }
            // Filter by category
            if (selectedCategoryId != null) {
                result = result.filter { it.categoryId == selectedCategoryId }
            }
            return result
        }
}

class HomeViewModel : ViewModel() {

    private val _uiState = MutableStateFlow(HomeUiState())
    val uiState: StateFlow<HomeUiState> = _uiState.asStateFlow()

    fun loadItems() {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(loadingState = LoadingState.Loading, error = null)
            try {
                val userId = FoodRepository.getCurrentUserId()
                if (userId != null) {
                    val items = FoodRepository.getFoodItems(userId)
                    val categories = FoodRepository.getCategories()
                    _uiState.value = HomeUiState(
                        foodItems = items,
                        categories = categories
                    )
                } else {
                    _uiState.value = HomeUiState(error = "Sesi tidak ditemukan")
                }
            } catch (e: Exception) {
                _uiState.value = _uiState.value.copy(
                    loadingState = LoadingState.Idle,
                    error = e.message ?: "Gagal memuat data"
                )
            }
        }
    }

    fun onSearchQueryChanged(query: String) {
        _uiState.value = _uiState.value.copy(searchQuery = query)
    }

    fun onCategorySelected(categoryId: String?) {
        _uiState.value = _uiState.value.copy(
            selectedCategoryId = if (_uiState.value.selectedCategoryId == categoryId) null
            else categoryId
        )
    }
}
