package com.example.foodiary2.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.foodiary2.data.FoodRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

data class AuthUiState(
    val email: String = "",
    val password: String = "",
    val loadingState: LoadingState = LoadingState.Idle,
    val isLoginMode: Boolean = true,
    val error: String? = null,
    val isSuccess: Boolean = false
) {
    val isLoading: Boolean get() = loadingState.isLoading
}

class AuthViewModel : ViewModel() {

    private val _uiState = MutableStateFlow(AuthUiState())
    val uiState: StateFlow<AuthUiState> = _uiState.asStateFlow()

    fun onEmailChanged(email: String) {
        _uiState.value = _uiState.value.copy(email = email, error = null)
    }

    fun onPasswordChanged(password: String) {
        _uiState.value = _uiState.value.copy(password = password, error = null)
    }

    fun toggleMode() {
        _uiState.value = _uiState.value.copy(
            isLoginMode = !_uiState.value.isLoginMode,
            error = null
        )
    }

    fun submit() {
        val state = _uiState.value
        if (state.email.isBlank() || state.password.isBlank()) {
            _uiState.value = state.copy(error = "Email dan password harus diisi")
            return
        }

        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(loadingState = LoadingState.Loading, error = null)
            try {
                if (state.isLoginMode) {
                    FoodRepository.signIn(state.email, state.password)
                } else {
                    FoodRepository.signUp(state.email, state.password)
                }

                // Verify session is actually active after auth
                val hasSession = FoodRepository.hasActiveSession()
                if (hasSession) {
                    _uiState.value = _uiState.value.copy(loadingState = LoadingState.Idle, isSuccess = true)
                } else {
                    // Signup succeeded but email confirmation is required
                    _uiState.value = _uiState.value.copy(
                        loadingState = LoadingState.Idle,
                        error = if (state.isLoginMode)
                            "Login gagal. Periksa kembali email dan password."
                        else
                            "Pendaftaran berhasil! Cek email untuk konfirmasi, lalu login."
                    )
                }
            } catch (e: Exception) {
                _uiState.value = _uiState.value.copy(
                    loadingState = LoadingState.Idle,
                    error = e.message ?: "Terjadi kesalahan"
                )
            }
        }
    }
}
