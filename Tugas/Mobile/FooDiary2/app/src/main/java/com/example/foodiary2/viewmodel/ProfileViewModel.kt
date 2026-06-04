package com.example.foodiary2.viewmodel

import android.content.ContentResolver
import android.net.Uri
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.foodiary2.data.FoodRepository
import com.example.foodiary2.model.Profile
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

data class ProfileUiState(
    val profile: Profile? = null,
    val email: String = "",
    val editName: String = "",
    val selectedAvatarUri: Uri? = null,
    val loadingState: LoadingState = LoadingState.Idle,
    val isSaving: Boolean = false,
    val saveSuccess: Boolean = false,
    val isLoggedOut: Boolean = false,
    val error: String? = null
) {
    val isLoading: Boolean get() = loadingState.isLoading
}

class ProfileViewModel : ViewModel() {

    private val _uiState = MutableStateFlow(ProfileUiState())
    val uiState: StateFlow<ProfileUiState> = _uiState.asStateFlow()

    private var selectedAvatarBytes: ByteArray? = null
    private var currentUserId: String? = null

    fun loadProfile() {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(loadingState = LoadingState.Loading)
            try {
                currentUserId = FoodRepository.getCurrentUserId()
                val email = FoodRepository.getCurrentUserEmail() ?: ""
                val profile = if (currentUserId != null) FoodRepository.getProfile(currentUserId!!) else null
                _uiState.value = ProfileUiState(
                    profile = profile,
                    email = email,
                    editName = profile?.fullName ?: ""
                )
            } catch (e: Exception) {
                _uiState.value = _uiState.value.copy(
                    loadingState = LoadingState.Idle,
                    error = e.message
                )
            }
        }
    }

    fun onNameChanged(name: String) {
        _uiState.value = _uiState.value.copy(editName = name, error = null)
    }

    fun onAvatarSelected(uri: Uri, contentResolver: ContentResolver) {
        selectedAvatarBytes = FoodRepository.readImageBytes(contentResolver, uri)
        _uiState.value = _uiState.value.copy(selectedAvatarUri = uri, error = null)
    }

    fun saveProfile() {
        val userId = currentUserId ?: return
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isSaving = true, error = null)
            try {
                var avatarUrl = _uiState.value.profile?.avatarUrl

                if (selectedAvatarBytes != null) {
                    val fileName = FoodRepository.generateAvatarFileName()
                    avatarUrl = FoodRepository.uploadAvatar(userId, selectedAvatarBytes!!, fileName)
                }

                val profile = Profile(
                    id = userId,
                    fullName = _uiState.value.editName,
                    avatarUrl = avatarUrl
                )

                FoodRepository.updateProfile(profile)
                _uiState.value = _uiState.value.copy(
                    isSaving = false,
                    saveSuccess = true,
                    profile = profile,
                    selectedAvatarUri = null
                )
                selectedAvatarBytes = null
            } catch (e: Exception) {
                _uiState.value = _uiState.value.copy(
                    isSaving = false,
                    error = e.message ?: "Gagal menyimpan profil"
                )
            }
        }
    }

    fun logout() {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(loadingState = LoadingState.Loading)
            try {
                FoodRepository.signOut()
                _uiState.value = _uiState.value.copy(loadingState = LoadingState.Idle, isLoggedOut = true)
            } catch (e: Exception) {
                _uiState.value = _uiState.value.copy(
                    loadingState = LoadingState.Idle,
                    error = e.message ?: "Gagal logout"
                )
            }
        }
    }
}
