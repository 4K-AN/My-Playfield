<?php

namespace App\Http\Controllers\Api\V1;

use App\Http\Controllers\Controller;
use App\Models\Container;
use App\Models\TrackingLog;
use Illuminate\Http\Request;
use Illuminate\Http\JsonResponse;
use OpenApi\Attributes as OA;

#[OA\Tag(name: 'Gateway - Tracking Logs', description: 'API Gateway untuk log perjalanan kontainer')]
class GatewayTrackingLogController extends Controller
{
    #[OA\Get(
        path: '/api/v1/gateway/containers/{containerId}/tracking-logs',
        operationId: 'getTrackingLogs',
        tags: ['Gateway - Tracking Logs'],
        summary: 'Ambil log perjalanan kontainer',
        description: 'Mengambil semua log perjalanan untuk kontainer tertentu.',
        security: [['bearerAuth' => []]],
        parameters: [
            new OA\Parameter(name: 'containerId', in: 'path', description: 'ID kontainer', required: true, schema: new OA\Schema(type: 'integer')),
        ],
        responses: [
            new OA\Response(response: 200, description: 'Log perjalanan berhasil diambil'),
            new OA\Response(response: 404, description: 'Kontainer tidak ditemukan'),
        ]
    )]
    public function index(int $containerId): JsonResponse
    {
        $container = Container::find($containerId);

        if (!$container) {
            return response()->json([
                'success' => false,
                'message' => 'Kontainer tidak ditemukan',
            ], 404);
        }

        $logs = $container->trackingLogs()->orderBy('logged_at', 'desc')->get();

        return response()->json([
            'success' => true,
            'data' => $logs,
        ]);
    }

    #[OA\Post(
        path: '/api/v1/gateway/containers/{containerId}/tracking-logs',
        operationId: 'createTrackingLog',
        tags: ['Gateway - Tracking Logs'],
        summary: 'Tambah log perjalanan (Admin only)',
        description: 'Membuat log perjalanan baru untuk kontainer tertentu. Hanya admin.',
        security: [['bearerAuth' => []]],
        parameters: [
            new OA\Parameter(name: 'containerId', in: 'path', description: 'ID kontainer', required: true, schema: new OA\Schema(type: 'integer')),
        ],
        requestBody: new OA\RequestBody(
            required: true,
            content: new OA\JsonContent(
                required: ['location_from', 'location_to'],
                properties: [
                    new OA\Property(property: 'location_from', type: 'string', example: 'Gudang Utama - Jakarta Utara'),
                    new OA\Property(property: 'location_to', type: 'string', example: 'Zona B - Tangerang'),
                    new OA\Property(property: 'status_change', type: 'string', example: 'Active → Full'),
                    new OA\Property(property: 'notes', type: 'string', example: 'Kontainer dipindahkan karena kapasitas penuh'),
                ]
            )
        ),
        responses: [
            new OA\Response(response: 201, description: 'Log berhasil ditambahkan'),
            new OA\Response(response: 403, description: 'Forbidden'),
            new OA\Response(response: 404, description: 'Kontainer tidak ditemukan'),
            new OA\Response(response: 422, description: 'Validasi gagal'),
        ]
    )]
    public function store(Request $request, int $containerId): JsonResponse
    {
        $container = Container::find($containerId);

        if (!$container) {
            return response()->json([
                'success' => false,
                'message' => 'Kontainer tidak ditemukan',
            ], 404);
        }

        $validated = $request->validate([
            'location_from' => 'required|string|max:255',
            'location_to' => 'required|string|max:255',
            'status_change' => 'nullable|string|max:255',
            'notes' => 'nullable|string',
        ]);

        $validated['container_id'] = $containerId;
        $validated['logged_at'] = now();

        $log = TrackingLog::create($validated);

        return response()->json([
            'success' => true,
            'message' => 'Log perjalanan berhasil ditambahkan',
            'data' => $log,
        ], 201);
    }
}
