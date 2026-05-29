<?php

namespace App\Http\Controllers;

use OpenApi\Attributes as OA;

#[OA\Info(
    version: '1.0.0',
    title: 'WowoClean API - Sistem Manajemen Kontainer Limbah B3',
    description: 'Enterprise REST API untuk manajemen kontainer limbah B3 dengan autentikasi JWT, otorisasi berbasis Role, dan API Gateway pattern.',
    contact: new OA\Contact(
        email: 'admin@wowoclean.com',
        name: 'WowoClean API Support'
    ),
    license: new OA\License(
        name: 'MIT',
        url: 'https://opensource.org/licenses/MIT'
    )
)]
#[OA\Server(
    url: '/',
    description: 'Local Development Server'
)]
#[OA\SecurityScheme(
    securityScheme: 'bearerAuth',
    type: 'http',
    scheme: 'bearer',
    bearerFormat: 'JWT',
    description: 'Masukkan JWT token yang didapat dari endpoint login.'
)]
abstract class Controller
{
    //
}
