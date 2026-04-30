<?php

namespace App\Http\Controllers\Api;

use App\Http\Controllers\Controller;
use App\Models\DummyUser;
use Illuminate\Http\Request;
use Tymon\JWTAuth\Exceptions\JWTException;
use Tymon\JWTAuth\Facades\JWTAuth;

class AuthController extends Controller
{
    // TUGAS 1: Menambahkan satu data user dummy baru
    private $users = [
        [
            'id' => 1,
            'name' => 'User Cakep',
            'email' => 'user@example.com',
            'password' => 'password123'
        ],
        [
            'id' => 2,
            'name' => 'Admin Hebat',
            'email' => 'admin@example.com',
            'password' => 'secret321'
        ],
        [
            'id' => 3,
            'name' => 'Akhmad Syafiul Anam',
            'email' => 'syafi@student.ub.ac.id',
            'password' => 'rahasia123'
        ]
    ];

    public function register(Request $request)
    {
        $validated = $this->validate($request, [
            'name' => 'required|string|max:100',
            'email' => 'required|email',
            'password' => 'required|string|min:6|confirmed'
        ]);

        // TUGAS 2: Validasi agar email harus unik terhadap daftar dummy yang sudah ada
        $isEmailExists = collect($this->users)->firstWhere('email', $validated['email']);
        
        if ($isEmailExists) {
            return response()->json([
                'message' => 'Pendaftaran gagal, email sudah terdaftar di sistem.'
            ], 422);
        }

        $user = [
            'id' => rand(4, 1000),
            'name' => $validated['name'],
            'email' => $validated['email'],
            'password' => $validated['password'],
        ];

        return response()->json([
            'message' => 'User registered successfully (dummy)',
            'user' => $user
        ], 201);
    }

    public function login(Request $request)
    {
        $credentials = $this->validate($request, [
            'email' => 'required|email',
            'password' => 'required|string'
        ]);

        $userData = collect($this->users)->firstWhere('email', $credentials['email']);

        if (!$userData || $userData['password'] !== $credentials['password']) {
            return response()->json([
                'message' => 'Invalid email or password'
            ], 401);
        }

        $user = new DummyUser($userData);
        $token = JWTAuth::claims([
            'email' => $user->email,
            'name' => $user->name
        ])->fromUser($user);

        return response()->json([
            'message' => 'Login successful (dummy)',
            'token' => $token
        ]);
    }

    public function logout()
    {
        try {
            JWTAuth::invalidate(JWTAuth::getToken());
            return response()->json([
                'message' => 'User logged out successfully'
            ]);
        } catch (JWTException $e) {
            return response()->json([
                'message' => 'Failed to logout, token invalid'
            ], 500);
        }
    }

    public function profile(Request $request)
    {
        try {
            $payload = $request->jwt_payload;
            return response()->json([
                'user' => [
                    'email' => $payload->get('email'),
                    'name' => $payload->get('name')
                ]
            ]);
        } catch (JWTException $e) {
            return response()->json([
                'message' => 'Token is invalid or expired'
            ], 401);
        }
    }

    // TUGAS 4: Membuat method untuk endpoint token-check
    public function tokenCheck(Request $request)
    {
        // Jika request berhasil masuk ke method ini, artinya token sudah pasti valid 
        // karena berhasil melewati DummyJwtMiddleware.
        $payload = $request->jwt_payload;
        
        return response()->json([
            'message' => 'Token valid',
            'user' => [
                'email' => $payload->get('email'),
                'name' => $payload->get('name')
            ]
        ], 200);
    }
}
