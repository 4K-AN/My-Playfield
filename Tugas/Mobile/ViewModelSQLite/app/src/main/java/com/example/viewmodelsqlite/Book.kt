package com.example.viewmodelsqlite

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "book")
data class Book(
    @PrimaryKey(autoGenerate = true)
    val id: Int? = null,
    val title: String,
    val isbn: String
)
