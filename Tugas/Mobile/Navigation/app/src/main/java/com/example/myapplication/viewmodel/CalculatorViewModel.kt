package com.example.myapplication.viewmodel

import androidx.lifecycle.ViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

class CalculatorViewModel : ViewModel() {
    private val _discount = MutableStateFlow(0)
    val discount: StateFlow<Int> = _discount

    fun calculateDiscount(amount: Int) {
        // Logika sederhana: Diskon 10%
        _discount.value = amount / 10
    }

    fun reset() {
        // SSoT: Reset data pusat di ViewModel
        _discount.value = 0
    }
}
