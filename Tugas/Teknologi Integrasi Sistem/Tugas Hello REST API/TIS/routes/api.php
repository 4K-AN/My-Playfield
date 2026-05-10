<?php

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Route;

Route::get('/ping', function () {
    return response()->json(['message' => 'pong']);
});

// a. Route Info
Route::get('/info', function () {
    return response()->json([
        "app" => "Todo API",
        "version" => "1.0",
        "developer" => "Teknologi Informasi"
    ]);
});
    
// b. Route User
// Route::get('/user/{name}', function ($name) {
//     return response()->json([
//         "message" => "Hello, " . $name . "!"
//     ]);
// });

// c. Route Calculator
Route::get('/calc/{a}/{b}/{op}', function ($a, $b, $op) {
    $numA = (float) $a;
    $numB = (float) $b;
    $result = 0;

    switch ($op) {
        case 'add': $result = $numA + $numB; break;
        case 'sub': $result = $numA - $numB; break;
        case 'mul': $result = $numA * $numB; break;
        case 'div': $result = ($numB != 0) ? $numA / $numB : "Error"; break;
        default: return response()->json(["error" => "Invalid operation"], 400);
    }

    return response()->json([
        "operation" => $op,
        "a" => $numA,
        "b" => $numB,
        "result" => $result
    ]);
});
// Route::get('/students/search', [App\Http\Controllers\StudentController::class, 'search']);
// d. Route Student CRUD
// Route::get('/students/search', [App\Http\Controllers\StudentController::class, 'search']);
Route::get('/students', [App\Http\Controllers\StudentController::class, 'index']);
Route::post('/students', [App\Http\Controllers\StudentController::class, 'store']);
Route::get('/students/{nim}', [App\Http\Controllers\StudentController::class, 'show']);
Route::put('/students/{nim}', [App\Http\Controllers\StudentController::class, 'update']);
Route::patch('/students/{nim}', [App\Http\Controllers\StudentController::class, 'update']);
Route::delete('/students/{nim}', [App\Http\Controllers\StudentController::class, 'destroy']);
Route::get('/students/{nim}/mata-kuliah', [App\Http\Controllers\StudentController::class, 'mataKuliahByStudent']);

// JWT Auth Routes
Route::post('/register', [\App\Http\Controllers\Api\AuthController::class, 'register']);
Route::post('/login', [\App\Http\Controllers\Api\AuthController::class, 'login']);

Route::middleware(['dummy.jwt'])->group(function() {
    Route::get('/profile', [\App\Http\Controllers\Api\AuthController::class, 'profile']);
    
    Route::get('/admin/dashboard', function() {
        return response()->json([
            'message' => 'Welcome to Admin Dashboard'
        ]);
    })->middleware('role:admin');
    
    Route::get('/user/dashboard', function() {
        return response()->json([
            'message' => 'Welcome to User Dashboard'
        ]);
    })->middleware('role:user');
    
    Route::get('/manager/dashboard', function() {
        return response()->json([
            'message' => 'Welcome to Manager Dashboard'
        ]);
    })->middleware('role:manager');
    
    Route::post('/logout', [\App\Http\Controllers\Api\AuthController::class, 'logout']);
});