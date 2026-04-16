<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Cache;
use Illuminate\Support\Facades\Validator;

class ContainerController extends Controller
{
    private function getContainers()
    {
        return Cache::rememberForever('containers', function () {
            return [
                [
                    'container_id' => 'WA12345',
                    'waste_type' => 'Plastic',
                    'weight_kg' => 500,
                    'status' => 'Active',
                    'tracking_logs' => [
                        [
                            'location' => 'Warehouse A',
                            'timestamp' => '2023-10-01T10:00:00',
                            'description' => 'Initial sorted'
                        ]
                    ]
                ],
                [
                    'container_id' => 'CH67890',
                    'waste_type' => 'Chemical',
                    'weight_kg' => 800,
                    'status' => 'Archived',
                    'tracking_logs' => [
                        [
                            'location' => 'Facility B',
                            'timestamp' => '2023-10-02T14:00:00',
                            'description' => 'Processed and safely stored'
                        ]
                    ]
                ]
            ];
        });
    }

    private function saveContainers($containers)
    {
        Cache::put('containers', $containers);
    }

    public function index()
    {
        return response()->json($this->getContainers());
    }

    public function search(Request $request)
    {
        $containers = $this->getContainers();
        $type = $request->query('type');
        $minWeight = $request->query('min_weight');

        $filtered = array_filter($containers, function ($item) use ($type, $minWeight) {
            $match = true;
            if ($type && strtolower($item['waste_type']) !== strtolower($type)) {
                $match = false;
            }
            if ($minWeight && $item['weight_kg'] < $minWeight) {
                $match = false;
            }
            return $match;
        });

        return response()->json(array_values($filtered));
    }

    public function store(Request $request)
    {
        $containers = $this->getContainers();
        $existingIds = array_column($containers, 'container_id');

        $validator = Validator::make($request->all(), [
            'container_id' => [
                'required',
                'string',
                'regex:/^[A-Za-z]{2}\d{5}$/',
                function ($attribute, $value, $fail) use ($existingIds) {
                    if (in_array($value, $existingIds)) {
                        $fail('The '.$attribute.' has already been taken.');
                    }
                },
            ],
            'waste_type' => 'required|string',
            'weight_kg' => 'required|numeric|min:10|max:5000',
        ]);

        $validator->after(function ($validator) use ($request) {
            if (strtolower($request->waste_type) === 'chemical' && $request->weight_kg > 1000) {
                $validator->errors()->add('weight_kg', 'Jika jenis chemical maka berat maksimal 1000.');
            }
        });

        if ($validator->fails()) {
            return response()->json(['errors' => $validator->errors()], 422);
        }

        $newContainer = [
            'container_id' => $request->container_id,
            'waste_type' => $request->waste_type,
            'weight_kg' => $request->weight_kg,
            'status' => 'Active',
            'tracking_logs' => [
                [
                    'location' => 'Origin',
                    'timestamp' => date('Y-m-d\TH:i:s'),
                    'description' => 'Container created'
                ]
            ]
        ];

        $containers[] = $newContainer;
        $this->saveContainers($containers);

        return response()->json($newContainer, 201);
    }

    public function updateStatus(Request $request, $id)
    {
        $containers = $this->getContainers();
        $found = false;

        foreach ($containers as &$container) {
            if ($container['container_id'] === $id) {
                $container['status'] = 'Archived';
                $found = true;
                break;
            }
        }

        if (!$found) {
            return response()->json(['message' => 'Not found'], 404);
        }

        $this->saveContainers($containers);
        return response()->json(['message' => 'Status updated to Archived']);
    }

    public function destroy($id)
    {
        $containers = $this->getContainers();
        $initialCount = count($containers);
        
        $containers = array_filter($containers, function ($item) use ($id) {
            return $item['container_id'] !== $id;
        });

        if (count($containers) === $initialCount) {
            return response()->json(['message' => 'Not found'], 404);
        }

        $this->saveContainers(array_values($containers));
        return response()->json(['message' => 'Container deleted']);
    }

    public function logs($id)
    {
        $containers = $this->getContainers();

        foreach ($containers as $container) {
            if ($container['container_id'] === $id) {
                return response()->json($container['tracking_logs']);
            }
        }

        return response()->json(['message' => 'Not found'], 404);
    }
}
