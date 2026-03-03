<?php

use Illuminate\Support\Facades\Route;
use App\Http\Controllers\HomeController; 

Route::get('/', function () {
    return "Welcome to Hello REST API!";
});

Route::get('/hello', function () {
    return "Hello Laravel!";
});

Route::get('/about', function () {
    return "Nama: Akhmad Syafiul Anam - NIM: [245150707111012]";
});

Route::get('/home', [HomeController::class, 'index']);