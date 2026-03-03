<?php

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Route;

Route::get('/status', function () {
    return response()->json([
        "app" => "Todo API",
        "status" => "running"
    ]);
});

Route::get('/greet/Akhmad-Syafiul', function () {
    return response()->json([
        "message" => "Hello, Akhmad Syafiul!"
    ]);
});

Route::get('/greet/{name}', function ($name) {
    return response()->json([
        "message" => "Hello, " . $name . "!"
    ]);
});