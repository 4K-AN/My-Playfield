<?php

use Illuminate\Support\Facades\Route;
use App\Http\Controllers\Api\V1\AuthController;
use App\Http\Controllers\Api\V1\GatewayContainerController;
use App\Http\Controllers\Api\V1\GatewayTrackingLogController;

/*
|--------------------------------------------------------------------------
| API Routes - Version 1
|--------------------------------------------------------------------------
|
| Semua route API menggunakan prefix /api/v1
| Gateway routes menggunakan prefix /api/v1/gateway
|
*/

Route::prefix('v1')->group(function () {

    // === Authentication Routes (Public) ===
    Route::post('/login', [AuthController::class, 'login']);

    // === Authenticated Routes ===
    Route::middleware('auth:api')->group(function () {

        // Auth
        Route::get('/profile', [AuthController::class, 'profile']);
        Route::post('/logout', [AuthController::class, 'logout']);

        // === API Gateway Routes ===
        Route::prefix('gateway')->group(function () {

            // Containers - Accessible by all authenticated users (GET)
            Route::get('/containers', [GatewayContainerController::class, 'index']);
            Route::get('/containers/{id}', [GatewayContainerController::class, 'show']);

            // Containers - Admin only (POST, PUT, PATCH, DELETE)
            Route::middleware('role:admin')->group(function () {
                Route::post('/containers', [GatewayContainerController::class, 'store']);
                Route::put('/containers/{id}', [GatewayContainerController::class, 'update']);
                Route::patch('/containers/{id}/archive', [GatewayContainerController::class, 'archive']);
                Route::delete('/containers/{id}', [GatewayContainerController::class, 'destroy']);
            });

            // Tracking Logs
            Route::get('/containers/{containerId}/tracking-logs', [GatewayTrackingLogController::class, 'index']);
            Route::middleware('role:admin')->group(function () {
                Route::post('/containers/{containerId}/tracking-logs', [GatewayTrackingLogController::class, 'store']);
            });
        });
    });
});
