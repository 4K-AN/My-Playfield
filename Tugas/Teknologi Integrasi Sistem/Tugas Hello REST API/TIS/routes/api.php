<?php

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Route;
use App\Http\Controllers\Api\GatewayController;

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
        case 'add':
            $result = $numA + $numB;
            break;
        case 'sub':
            $result = $numA - $numB;
            break;
        case 'mul':
            $result = $numA * $numB;
            break;
        case 'div':
            $result = ($numB != 0) ? $numA / $numB : "Error";
            break;
        default:
            return response()->json(["error" => "Invalid operation"], 400);
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
Route::get('/students/{nim}/mata-kuliah', [App\Http\Controllers\StudentController::class, 'coursesByStudent']);

// JWT Auth Routes
Route::post('/register', [\App\Http\Controllers\Api\AuthController::class, 'register']);
Route::post('/login', [\App\Http\Controllers\Api\AuthController::class, 'login']);

Route::middleware(['dummy.jwt'])->group(function () {
    Route::get('/profile', [\App\Http\Controllers\Api\GatewayController::class, 'getProfile'])
         ->middleware('role:admin,user,manager');

    Route::get('/admin/dashboard', function () {
        return response()->json([
            'message' => 'Welcome to Admin Dashboard'
        ]);
    })->middleware('role:admin');

    Route::get('/user/dashboard', function () {
        return response()->json([
            'message' => 'Welcome to User Dashboard'
        ]);
    })->middleware('role:user');

    Route::get('/manager/dashboard', function () {
        return response()->json([
            'message' => 'Welcome to Manager Dashboard'
        ]);
    })->middleware('role:manager');

    Route::post('/logout', [\App\Http\Controllers\Api\AuthController::class, 'logout']);
});

// API Gateway Routes
Route::middleware(['dummy.jwt'])->prefix('gateway')->group(function () {
    // Profile endpoint
    Route::get('/profile', [GatewayController::class, 'getProfile'])
        ->middleware('role:admin,user,manager');
    
    // Student CRUD endpoints
    Route::get('/students', [GatewayController::class, 'getStudents'])
        ->middleware('role:admin,user,manager');
    Route::post('/students', [GatewayController::class, 'createStudent'])
        ->middleware('role:admin');
    Route::put('/students/{nim}', [GatewayController::class, 'updateStudent'])
        ->middleware('role:admin');
    Route::patch('/students/{nim}', [GatewayController::class, 'updateStudent'])
        ->middleware('role:admin');
    Route::delete('/students/{nim}', [GatewayController::class, 'deleteStudent'])
        ->middleware('role:admin');
    
    // Dashboard endpoints
    Route::get('/admin/dashboard', [GatewayController::class, 'getAdminDashboard'])
        ->middleware('role:admin');
    Route::get('/user/dashboard', [GatewayController::class, 'getUserDashboard'])
        ->middleware('role:user,manager');
});