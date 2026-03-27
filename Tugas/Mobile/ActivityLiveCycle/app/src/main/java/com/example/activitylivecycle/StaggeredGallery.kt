package com.example.activitylivecycle

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.staggeredgrid.LazyVerticalStaggeredGrid
import androidx.compose.foundation.lazy.staggeredgrid.StaggeredGridCells
import androidx.compose.foundation.lazy.staggeredgrid.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.unit.dp
import coil.compose.AsyncImage

// Generate 30 link gambar secara otomatis dengan tinggi acak
val fotoGaleri = List(30) { "https://picsum.photos/300/${(250..550).random()}" }

@Composable
fun StaggeredGalleryExample() {
    LazyVerticalStaggeredGrid(
        // Menggunakan 3 kolom
        columns = StaggeredGridCells.Fixed(3), 
        modifier = Modifier.fillMaxSize(),
        contentPadding = PaddingValues(16.dp),
        verticalItemSpacing = 12.dp,
        horizontalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        items(fotoGaleri) { urlGambar ->
            AsyncImage(
                model = urlGambar,
                contentDescription = "Gambar Staggered",
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(16.dp)),
                contentScale = ContentScale.FillWidth
            )
        }
    }
}