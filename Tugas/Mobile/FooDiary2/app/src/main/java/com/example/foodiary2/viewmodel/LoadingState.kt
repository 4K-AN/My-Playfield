package com.example.foodiary2.viewmodel

/**
 * Loading state yang lebih detail untuk membedakan jenis operasi.
 */
sealed class LoadingState {
    /** Tidak ada operasi yang berjalan. */
    data object Idle : LoadingState()

    /** Operasi loading data utama. */
    data object Loading : LoadingState()

    /** Operasi menyimpan data. */
    data object Saving : LoadingState()

    /** Operasi menghapus data. */
    data object Deleting : LoadingState()

    val isLoading: Boolean get() = this is Loading || this is Saving || this is Deleting
}
