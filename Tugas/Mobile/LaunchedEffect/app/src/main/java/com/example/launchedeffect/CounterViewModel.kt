package com.example.launchedeffect

import androidx.lifecycle.ViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update

class CounterViewModel : ViewModel() {
    private val _angka = MutableStateFlow(0)
    val angka: StateFlow<Int> = _angka.asStateFlow()

    fun tambahAngka() {
        _angka.update { nilaiLama ->
            nilaiLama + 1
        }
    }
}
