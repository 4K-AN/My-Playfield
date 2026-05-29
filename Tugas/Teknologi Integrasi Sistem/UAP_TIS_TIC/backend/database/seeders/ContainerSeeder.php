<?php

namespace Database\Seeders;

use App\Models\Container;
use Illuminate\Database\Seeder;

class ContainerSeeder extends Seeder
{
    /**
     * Run the database seeds.
     */
    public function run(): void
    {
        $containers = [
            [
                'container_code' => 'WC-LC-001',
                'type' => 'limbah_cair',
                'capacity' => 5000.00,
                'current_fill_level' => 3200.50,
                'location' => 'Gudang Utama - Jakarta Utara',
                'status' => 'Active',
                'last_maintenance_date' => '2024-11-15',
            ],
            [
                'container_code' => 'WC-LP-002',
                'type' => 'limbah_padat',
                'capacity' => 8000.00,
                'current_fill_level' => 7800.00,
                'location' => 'Zona B - Tangerang',
                'status' => 'Full',
                'last_maintenance_date' => '2024-10-20',
            ],
            [
                'container_code' => 'WC-LG-003',
                'type' => 'limbah_gas',
                'capacity' => 3000.00,
                'current_fill_level' => 1500.00,
                'location' => 'Area Kimia - Bekasi',
                'status' => 'Active',
                'last_maintenance_date' => '2024-12-01',
            ],
            [
                'container_code' => 'WC-LC-004',
                'type' => 'limbah_cair',
                'capacity' => 6000.00,
                'current_fill_level' => 0.00,
                'location' => 'Depo Sementara - Karawang',
                'status' => 'Maintenance',
                'last_maintenance_date' => '2024-12-10',
            ],
            [
                'container_code' => 'WC-LP-005',
                'type' => 'limbah_padat',
                'capacity' => 10000.00,
                'current_fill_level' => 4500.00,
                'location' => 'Gudang Utama - Jakarta Utara',
                'status' => 'Active',
                'last_maintenance_date' => '2024-09-30',
            ],
            [
                'container_code' => 'WC-LG-006',
                'type' => 'limbah_gas',
                'capacity' => 2000.00,
                'current_fill_level' => 1900.00,
                'location' => 'Area Kimia - Bekasi',
                'status' => 'Full',
                'last_maintenance_date' => '2024-11-05',
            ],
            [
                'container_code' => 'WC-LC-007',
                'type' => 'limbah_cair',
                'capacity' => 4000.00,
                'current_fill_level' => 2000.00,
                'location' => 'Zona C - Cikarang',
                'status' => 'Active',
                'last_maintenance_date' => '2024-12-15',
            ],
            [
                'container_code' => 'WC-LP-008',
                'type' => 'limbah_padat',
                'capacity' => 7000.00,
                'current_fill_level' => 0.00,
                'location' => 'Depo Sementara - Karawang',
                'status' => 'Archived',
                'last_maintenance_date' => '2024-08-20',
            ],
        ];

        foreach ($containers as $container) {
            Container::create($container);
        }
    }
}
