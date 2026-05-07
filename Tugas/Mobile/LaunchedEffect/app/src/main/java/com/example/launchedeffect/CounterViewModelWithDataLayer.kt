package com.example.launchedeffect

import androidx.lifecycle.ViewModel
import kotlinx.coroutines.flow.StateFlow

class CounterViewModelWithDataLayer(
    private val repository: CounterRepository = CounterRepository()
) : ViewModel() {
    val angka: StateFlow<Int> = repository.angka
    
    fun tambahAngka() {
        repository.tambahAngka()
    }
}
