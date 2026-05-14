<?php

namespace App\Http\Controllers\Api;

use App\Http\Controllers\Controller;
use App\Http\Controllers\StudentController;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Log;
use Tymon\JWTAuth\Facades\JWTAuth;

class GatewayController extends Controller
{
    /**
     * Get all students through gateway
     * GET /api/gateway/students
     */
    public function getStudents(Request $request)
    {
        try {
            // Log the request
            $payload = $request->jwt_payload;
            Log::info('Gateway: GET /students', [
                'user' => $payload->get('email'),
                'role' => $payload->get('role'),
                'method' => 'GET',
                'endpoint' => '/gateway/students'
            ]);

            $studentController = new StudentController();
            $response = $studentController->index();
            
            return response()->json([
                'gateway' => 'API Gateway',
                'message' => 'Request forwarded to Student Service',
                'result' => $response->getData()
            ]);
        } catch (\Exception $e) {
            Log::error('Gateway Error: GET /students', [
                'error' => $e->getMessage()
            ]);
            return response()->json([
                'message' => 'Error processing request',
                'error' => $e->getMessage()
            ], 500);
        }
    }

    /**
     * Create a new student through gateway
     * POST /api/gateway/students
     */
    public function createStudent(Request $request)
    {
        try {
            // Log the request
            $payload = $request->jwt_payload;
            Log::info('Gateway: POST /students', [
                'user' => $payload->get('email'),
                'role' => $payload->get('role'),
                'method' => 'POST',
                'endpoint' => '/gateway/students'
            ]);

            $studentController = new StudentController();
            return $studentController->store($request);
        } catch (\Exception $e) {
            Log::error('Gateway Error: POST /students', [
                'error' => $e->getMessage()
            ]);
            return response()->json([
                'message' => 'Error processing request',
                'error' => $e->getMessage()
            ], 500);
        }
    }

    /**
     * Update a student through gateway
     * PUT /api/gateway/students/{nim}
     */
    public function updateStudent(Request $request, $nim)
    {
        try {
            // Log the request
            $payload = $request->jwt_payload;
            Log::info('Gateway: PUT /students/{nim}', [
                'user' => $payload->get('email'),
                'role' => $payload->get('role'),
                'method' => 'PUT',
                'endpoint' => '/gateway/students/' . $nim
            ]);

            $studentController = new StudentController();
            return $studentController->update($request, $nim);
        } catch (\Exception $e) {
            Log::error('Gateway Error: PUT /students/{nim}', [
                'error' => $e->getMessage()
            ]);
            return response()->json([
                'message' => 'Error processing request',
                'error' => $e->getMessage()
            ], 500);
        }
    }

    /**
     * Delete a student through gateway
     * DELETE /api/gateway/students/{nim}
     */
    public function deleteStudent(Request $request, $nim)
    {
        try {
            // Log the request
            $payload = $request->jwt_payload;
            Log::info('Gateway: DELETE /students/{nim}', [
                'user' => $payload->get('email'),
                'role' => $payload->get('role'),
                'method' => 'DELETE',
                'endpoint' => '/gateway/students/' . $nim
            ]);

            $studentController = new StudentController();
            return $studentController->destroy($nim);
        } catch (\Exception $e) {
            Log::error('Gateway Error: DELETE /students/{nim}', [
                'error' => $e->getMessage()
            ]);
            return response()->json([
                'message' => 'Error processing request',
                'error' => $e->getMessage()
            ], 500);
        }
    }

    /**
     * Get user profile through gateway
     * GET /api/gateway/profile
     */
    public function getProfile(Request $request)
    {
        try {
            // Log the request
            $payload = $request->jwt_payload;
            Log::info('Gateway: GET /profile', [
                'user' => $payload->get('email'),
                'role' => $payload->get('role'),
                'method' => 'GET',
                'endpoint' => '/gateway/profile'
            ]);

            $authController = new AuthController();
            $profileResponse = $authController->profile($request);
            
            return response()->json([
                'gateway' => 'API Gateway',
                'message' => 'Request forwarded to Auth Service (Profile)',
                'result' => $profileResponse->original
            ]);
        } catch (\Exception $e) {
            Log::error('Gateway Error: GET /profile', [
                'error' => $e->getMessage()
            ]);
            return response()->json([
                'message' => 'Error processing request',
                'error' => $e->getMessage()
            ], 500);
        }
    }

    /**
     * Get admin dashboard through gateway
     * GET /api/gateway/admin/dashboard
     */
    public function getAdminDashboard(Request $request)
    {
        try {
            // Log the request
            $payload = $request->jwt_payload;
            Log::info('Gateway: GET /admin/dashboard', [
                'user' => $payload->get('email'),
                'role' => $payload->get('role'),
                'method' => 'GET',
                'endpoint' => '/gateway/admin/dashboard'
            ]);

            return response()->json([
                'gateway' => 'API Gateway',
                'message' => 'Welcome to Admin Dashboard',
                'user' => [
                    'email' => $payload->get('email'),
                    'name' => $payload->get('name'),
                    'role' => $payload->get('role')
                ]
            ]);
        } catch (\Exception $e) {
            Log::error('Gateway Error: GET /admin/dashboard', [
                'error' => $e->getMessage()
            ]);
            return response()->json([
                'message' => 'Error processing request',
                'error' => $e->getMessage()
            ], 500);
        }
    }

    /**
     * Get user dashboard through gateway
     * GET /api/gateway/user/dashboard
     */
    public function getUserDashboard(Request $request)
    {
        try {
            // Log the request
            $payload = $request->jwt_payload;
            Log::info('Gateway: GET /user/dashboard', [
                'user' => $payload->get('email'),
                'role' => $payload->get('role'),
                'method' => 'GET',
                'endpoint' => '/gateway/user/dashboard'
            ]);

            return response()->json([
                'gateway' => 'API Gateway',
                'message' => 'Welcome to User Dashboard',
                'user' => [
                    'email' => $payload->get('email'),
                    'name' => $payload->get('name'),
                    'role' => $payload->get('role')
                ]
            ]);
        } catch (\Exception $e) {
            Log::error('Gateway Error: GET /user/dashboard', [
                'error' => $e->getMessage()
            ]);
            return response()->json([
                'message' => 'Error processing request',
                'error' => $e->getMessage()
            ], 500);
        }
    }
}
