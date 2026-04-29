<?php

return [

    /*
    |--------------------------------------------------------------------------
    | JWT Authentication Secret
    |--------------------------------------------------------------------------
    |
    | Don't forget to set this in your .env file, as it will be used to sign
    | your tokens. A helper command is provided for this:
    | `php artisan jwt:secret`
    |
    | Note: This will be used for Symmetric signing
    |
    */

    'secret' => env('JWT_SECRET'),

    /*
    |--------------------------------------------------------------------------
    | JWT Authentication Keys
    |--------------------------------------------------------------------------
    |
    | The algorithm you want to use for token signing you can visit laravel
    | jwt authentication to find the valid algorithm for this class
    |
    | Supported algorithms by env('JWT_ALGORITHM'):
    | HS256, HS384, HS512, RS256, RS384, RS512, ES256, ES384, ES512
    |
    */

    'keys' => [
        'public' => env('JWT_PUBLIC_KEY'),
        'private' => env('JWT_PRIVATE_KEY'),
    ],

    /*
    |--------------------------------------------------------------------------
    | Payload
    |--------------------------------------------------------------------------
    |
    | Configure some specs of the curent jwt instance working on the tokens
    |
    */

    'supported_algs' => [
        'HS256',
        'HS384',
        'HS512',
        'RS256',
        'RS384',
        'RS512',
        'ES256',
        'ES384',
        'ES512',
    ],

    'algorithm' => env('JWT_ALGORITHM', 'HS256'),

    'payload' => [

        /*
        |--------------------------------------------------------------------------
        | Payload Claims
        |--------------------------------------------------------------------------
        |
        | When adding these claims to the payload, they will be hidden behind
        | protected claims since they are considered sensitive, you can add
        | as many claims as you want depending on 'supported_claims' above.
        |
        | Supports (deprecated): `Lcobucci\JWT\Signer\Key\InMemory` instance from `lcobucci/jwt: 3.3`, but not recommended
        |
        */

        'sub' => env('JWT_SUBJECT'),
        'iss' => env('JWT_ISSUER'),
        'aud' => env('JWT_AUDIENCE'),
        'iat' => true,
        'exp' => true,

    ],

    /*
    |--------------------------------------------------------------------------
    | Blacklist Storage
    |--------------------------------------------------------------------------
    |
    | This is the storage configuration for storing revoked tokens. After
    | every token revocation, a unique identifier will be stored in the
    | cache, in order to prevent using the token a second time.
    |
    */

    'blacklist_enabled' => env('JWT_BLACKLIST_ENABLED', true),

    'blacklist_storage' => env('JWT_BLACKLIST_STORAGE', 'cache'),

    'blacklist_cache' => env('JWT_BLACKLIST_CACHE', 'default'),

];
