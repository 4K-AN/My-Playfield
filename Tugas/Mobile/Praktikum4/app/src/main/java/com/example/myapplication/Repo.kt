package com.example.myapplication

import kotlinx.coroutines.delay
import kotlin.random.Random

class Repo {
    companion object {
        suspend fun getData() : Int {
            delay(2000) // Simulating network
            return Random.nextInt(100, 1000)
        }
    }
}
