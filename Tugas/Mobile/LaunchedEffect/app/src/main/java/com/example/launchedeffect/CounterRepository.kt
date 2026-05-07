package com.example.launchedeffect

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update

class CounterRepository {
    private val _angka = MutableStateFlow(0)
    val angka: StateFlow<Int> = _angka.asStateFlow()
    
    fun tambahAngka() {
        _angka.update { it + 1 }
    }
}
