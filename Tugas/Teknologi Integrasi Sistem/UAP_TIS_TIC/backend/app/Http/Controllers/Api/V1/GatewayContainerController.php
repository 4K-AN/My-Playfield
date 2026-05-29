<?php

namespace App\Http\Controllers\Api\V1;

use App\Http\Controllers\Controller;
use App\Http\Requests\StoreContainerRequest;
use App\Http\Requests\UpdateContainerRequest;
use App\Models\Container;
use Illuminate\Http\Request;
use Illuminate\Http\JsonResponse;
use OpenApi\Attributes as OA;

#[OA\Tag(name: 'Gateway - Containers', description: 'API Gateway untuk manajemen kontainer limbah B3')]
class GatewayContainerController extends Controller
{
    #[OA\Get(
        path: '/api/v1/gateway/containers',
        operationId: 'getContainers',
        tags: ['Gateway - Containers'],
        summary: 'Ambil daftar semua kontainer',
        description: 'Mengambil semua kontainer dengan fitur pencarian dan filter. Dapat diakses oleh semua role (admin & user).',
        security: [['bearerAuth' => []]],
        parameters: [
            new OA\Parameter(name: 'search', in: 'query', description: 'Pencarian berdasarkan kode kontainer atau lokasi', required: false, schema: new OA\Schema(type: 'string', example: 'WC-LC')),
            new OA\Parameter(name: 'type', in: 'query', description: 'Filter berdasarkan tipe limbah', required: false, schema: new OA\Schema(type: 'string', enum: ['limbah_cair', 'limbah_padat', 'limbah_gas'])),
            new OA\Parameter(name: 'status', in: 'query', description: 'Filter berdasarkan status kontainer', required: false, schema: new OA\Schema(type: 'string', enum: ['Active', 'Maintenance', 'Archived', 'Full'])),
        ],
        responses: [
            new OA\Response(
                response: 200,
                description: 'Daftar kontainer berhasil diambil',
                content: new OA\JsonContent(
                    properties: [
                        new OA\Property(property: 'success', type: 'boolean', example: true),
                        new OA\Property(property: 'message', type: 'string', example: 'Data kontainer berhasil diambil'),
                        new OA\Property(property: 'data', type: 'array', items: new OA\Items(type: 'object')),
                    ]
                )
            ),
            new OA\Response(response: 401, description: 'Unauthenticated'),
        ]
    )]
    public function index(Request $request): JsonResponse
    {
        $query = Container::query();

        // Pencarian berdasarkan kode kontainer atau lokasi
        if ($request->has('search') && $request->search !== '') {
            $search = $request->search;
            $query->where(function ($q) use ($search) {
                $q->where('container_code', 'like', "%{$search}%")
                  ->orWhere('location', 'like', "%{$search}%");
            });
        }

        // Filter berdasarkan tipe
        if ($request->has('type') && $request->type !== '') {
            $query->where('type', $request->type);
        }

        // Filter berdasarkan status
        if ($request->has('status') && $request->status !== '') {
            $query->where('status', $request->status);
        }

        $containers = $query->orderBy('created_at', 'desc')->get();

        return response()->json([
            'success' => true,
            'message' => 'Data kontainer berhasil diambil',
            'data' => $containers,
        ]);
    }

    #[OA\Get(
        path: '/api/v1/gateway/containers/{id}',
        operationId: 'getContainerById',
        tags: ['Gateway - Containers'],
        summary: 'Ambil detail kontainer berdasarkan ID',
        description: 'Mengambil detail kontainer beserta tracking logs. Dapat diakses oleh semua role.',
        security: [['bearerAuth' => []]],
        parameters: [
            new OA\Parameter(name: 'id', in: 'path', description: 'ID kontainer', required: true, schema: new OA\Schema(type: 'integer')),
        ],
        responses: [
            new OA\Response(response: 200, description: 'Detail kontainer berhasil diambil'),
            new OA\Response(response: 404, description: 'Kontainer tidak ditemukan'),
        ]
    )]
    public function show(int $id): JsonResponse
    {
        $container = Container::with('trackingLogs')->find($id);

        if (!$container) {
            return response()->json([
                'success' => false,
                'message' => 'Kontainer tidak ditemukan',
            ], 404);
        }

        return response()->json([
            'success' => true,
            'data' => $container,
        ]);
    }

    #[OA\Post(
        path: '/api/v1/gateway/containers',
        operationId: 'createContainer',
        tags: ['Gateway - Containers'],
        summary: 'Buat kontainer baru (Admin only)',
        description: 'Membuat kontainer limbah B3 baru. Hanya dapat diakses oleh role admin.',
        security: [['bearerAuth' => []]],
        requestBody: new OA\RequestBody(
            required: true,
            content: new OA\JsonContent(
                required: ['container_code', 'type', 'capacity', 'location'],
                properties: [
                    new OA\Property(property: 'container_code', type: 'string', example: 'WC-LC-009'),
                    new OA\Property(property: 'type', type: 'string', enum: ['limbah_cair', 'limbah_padat', 'limbah_gas'], example: 'limbah_cair'),
                    new OA\Property(property: 'capacity', type: 'number', example: 5000),
                    new OA\Property(property: 'current_fill_level', type: 'number', example: 0),
                    new OA\Property(property: 'location', type: 'string', example: 'Gudang Baru - Surabaya'),
                    new OA\Property(property: 'status', type: 'string', enum: ['Active', 'Maintenance', 'Archived', 'Full'], example: 'Active'),
                    new OA\Property(property: 'last_maintenance_date', type: 'string', format: 'date', example: '2024-12-20'),
                ]
            )
        ),
        responses: [
            new OA\Response(response: 201, description: 'Kontainer berhasil dibuat'),
            new OA\Response(response: 403, description: 'Forbidden - Hanya admin'),
            new OA\Response(response: 422, description: 'Validasi gagal'),
        ]
    )]
    public function store(StoreContainerRequest $request): JsonResponse
    {
        $container = Container::create($request->validated());

        return response()->json([
            'success' => true,
            'message' => 'Kontainer berhasil dibuat',
            'data' => $container,
        ], 201);
    }

    #[OA\Put(
        path: '/api/v1/gateway/containers/{id}',
        operationId: 'updateContainer',
        tags: ['Gateway - Containers'],
        summary: 'Update kontainer (Admin only)',
        description: 'Memperbarui data kontainer limbah B3. Hanya dapat diakses oleh role admin.',
        security: [['bearerAuth' => []]],
        parameters: [
            new OA\Parameter(name: 'id', in: 'path', description: 'ID kontainer', required: true, schema: new OA\Schema(type: 'integer')),
        ],
        requestBody: new OA\RequestBody(
            required: true,
            content: new OA\JsonContent(
                properties: [
                    new OA\Property(property: 'container_code', type: 'string'),
                    new OA\Property(property: 'type', type: 'string', enum: ['limbah_cair', 'limbah_padat', 'limbah_gas']),
                    new OA\Property(property: 'capacity', type: 'number'),
                    new OA\Property(property: 'current_fill_level', type: 'number'),
                    new OA\Property(property: 'location', type: 'string'),
                    new OA\Property(property: 'status', type: 'string', enum: ['Active', 'Maintenance', 'Archived', 'Full']),
                    new OA\Property(property: 'last_maintenance_date', type: 'string', format: 'date'),
                ]
            )
        ),
        responses: [
            new OA\Response(response: 200, description: 'Kontainer berhasil diperbarui'),
            new OA\Response(response: 403, description: 'Forbidden'),
            new OA\Response(response: 404, description: 'Kontainer tidak ditemukan'),
            new OA\Response(response: 422, description: 'Validasi gagal'),
        ]
    )]
    public function update(UpdateContainerRequest $request, int $id): JsonResponse
    {
        $container = Container::find($id);

        if (!$container) {
            return response()->json([
                'success' => false,
                'message' => 'Kontainer tidak ditemukan',
            ], 404);
        }

        $container->update($request->validated());

        return response()->json([
            'success' => true,
            'message' => 'Kontainer berhasil diperbarui',
            'data' => $container->fresh(),
        ]);
    }

    #[OA\Patch(
        path: '/api/v1/gateway/containers/{id}/archive',
        operationId: 'archiveContainer',
        tags: ['Gateway - Containers'],
        summary: 'Archive kontainer (Admin only)',
        description: 'Mengubah status kontainer menjadi Archived. Hanya admin.',
        security: [['bearerAuth' => []]],
        parameters: [
            new OA\Parameter(name: 'id', in: 'path', description: 'ID kontainer', required: true, schema: new OA\Schema(type: 'integer')),
        ],
        responses: [
            new OA\Response(response: 200, description: 'Kontainer berhasil di-archive'),
            new OA\Response(response: 403, description: 'Forbidden'),
            new OA\Response(response: 404, description: 'Kontainer tidak ditemukan'),
        ]
    )]
    public function archive(int $id): JsonResponse
    {
        $container = Container::find($id);

        if (!$container) {
            return response()->json([
                'success' => false,
                'message' => 'Kontainer tidak ditemukan',
            ], 404);
        }

        $container->update(['status' => 'Archived']);

        return response()->json([
            'success' => true,
            'message' => 'Kontainer berhasil di-archive',
            'data' => $container->fresh(),
        ]);
    }

    #[OA\Delete(
        path: '/api/v1/gateway/containers/{id}',
        operationId: 'deleteContainer',
        tags: ['Gateway - Containers'],
        summary: 'Hapus kontainer (Admin only)',
        description: 'Menghapus kontainer dari database. Hanya admin.',
        security: [['bearerAuth' => []]],
        parameters: [
            new OA\Parameter(name: 'id', in: 'path', description: 'ID kontainer', required: true, schema: new OA\Schema(type: 'integer')),
        ],
        responses: [
            new OA\Response(response: 200, description: 'Kontainer berhasil dihapus'),
            new OA\Response(response: 403, description: 'Forbidden'),
            new OA\Response(response: 404, description: 'Kontainer tidak ditemukan'),
        ]
    )]
    public function destroy(int $id): JsonResponse
    {
        $container = Container::find($id);

        if (!$container) {
            return response()->json([
                'success' => false,
                'message' => 'Kontainer tidak ditemukan',
            ], 404);
        }

        $container->delete();

        return response()->json([
            'success' => true,
            'message' => 'Kontainer berhasil dihapus',
        ]);
    }
}
