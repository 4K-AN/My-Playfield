<?php

/** @var \Laravel\Lumen\Routing\Router $router */
/** @var \Laravel\Lumen\Routing\Router $router */

use App\Http\Controllers\Api\AuthController;

$router->get('/ping', function () {
    return response()->json(['message' => 'pong']);
});

// JWT Authentication Routes
$router->post('/register', 'Api\AuthController@register');
$router->post('/login', 'Api\AuthController@login');

// Endpoint yang diproteksi oleh JWT
$router->group(['middleware' => 'dummy.jwt'], function () use ($router) {
    $router->post('/logout', 'Api\AuthController@logout');
    $router->get('/profile', 'Api\AuthController@profile');
    
    // Tugas 3: Menambahkan endpoint token-check
    $router->get('/token-check', 'Api\AuthController@tokenCheck');
});
