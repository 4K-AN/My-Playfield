<?php

namespace App\Http\Middleware;

use Closure;
use Illuminate\Http\Request;
use Symfony\Component\HttpFoundation\Response;
use Tymon\JWTAuth\Facades\JWTAuth;
use Exception;

class DummyJwtMiddleware
{
    public function handle(Request $request, Closure $next): Response
    {
        try {
            $payload = JWTAuth::parseToken()->getPayload();
            $request->jwt_payload = $payload;
        } catch (Exception $e) {
            return response()->json([
                'message' => 'Token is invalid or missing'
            ], 401);
        }

        return $next($request);
    }
}
