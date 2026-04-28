package com.example.myapplication.viewmodel

import androidx.lifecycle.ViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update

data class User(
    val name: String = "Guest",
    val status: String = "Non-Member"
)

class CalculatorViewModel : ViewModel() {
    private val _discount = MutableStateFlow(0)
    val discount: StateFlow<Int> = _discount

    private val _user = MutableStateFlow(User())
    val user: StateFlow<User> = _user

    fun compute(number: Int) {
        if (number > 100000) {
            val percent = 0.2
            _discount.update { (percent * number).toInt() }
            setMemberName("Akhmad Syafiul")
            setMemberStatus(true)
        } else {
            val percent = 0.1
            _discount.update { (percent * number).toInt() }
        }
    }

    private fun setMemberName(newName: String) {
        _user.update { it.copy(name = newName) }
    }

    private fun setMemberStatus(isPremium: Boolean) {
        _user.update { it.copy(status = if (isPremium) "Premium Member" else "Non-Member") }
    }

    fun reset() {
        _discount.update { 0 }
        setMemberName("Guest")
        setMemberStatus(false)
    }
}
