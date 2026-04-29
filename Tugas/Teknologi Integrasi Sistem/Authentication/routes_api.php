<?php

use App\Http\Controllers\Api\AuthController;
use App\Http\Controllers\Api\StudentController;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Route;

Route::get('user', function (Request $request) {
    return $request->user();
})->middleware('auth:sanctum');

Route::get('ping', function () {
    return response()->json(['message' => 'pong']);
});

// CRUD Student
Route::post('students', [StudentController::class, 'store']);
Route::get('students', [StudentController::class, 'index']);
Route::put('students/{nim}', [StudentController::class, 'update']);
Route::patch('students/{nim}', [StudentController::class, 'update']);
Route::delete('students/{nim}', [StudentController::class, 'destroy']);

// JWT Authentication
Route::post('register', [AuthController::class, 'register']);
Route::post('login', [AuthController::class, 'login']);

Route::middleware(['dummy.jwt'])->group(function () {
    Route::post('logout', [AuthController::class, 'logout']);
    Route::get('profile', [AuthController::class, 'profile']);
});
