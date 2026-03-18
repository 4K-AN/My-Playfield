<?php

use Illuminate\Support\Facades\Route;
use App\Http\Controllers\HomeController;
use App\Http\Controllers\DashboardController;

Route::get('/', [HomeController::class, 'index']);

// a. Route Profile
Route::get('/profile', function () {
    return "Nama: Akhmad Syafiul Anam\nNIM: 245150707111012\nProgram Studi: Teknologi Informasi";
});

// b. Route Welcome dengan parameter dinamis
Route::get('/welcome/{name}', function ($name) {
    return "Selamat datang, " . $name . "!";
});

// c. Route Dashboard ke Controller
Route::get('/dashboard', [DashboardController::class, 'index']);