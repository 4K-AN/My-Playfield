<?php

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Route;
use App\Http\Controllers\StudentController;

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

// Student API Routes (CRUD Operations)
Route::get('/students', [StudentController::class, 'index']);              // READ - Get all students
Route::post('/students', [StudentController::class, 'store']);             // CREATE - Create a new student
Route::put('/students/{nim}', [StudentController::class, 'update']);       // UPDATE - Update student (full)
Route::patch('/students/{nim}', [StudentController::class, 'update']);     // UPDATE - Update student (partial)
Route::delete('/students/{nim}', [StudentController::class, 'destroy']);   // DELETE - Delete a student