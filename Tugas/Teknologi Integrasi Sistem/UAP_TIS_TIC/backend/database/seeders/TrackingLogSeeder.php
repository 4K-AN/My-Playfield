<?php

namespace Database\Seeders;

use App\Models\TrackingLog;
use App\Models\Container;
use Illuminate\Database\Seeder;

class TrackingLogSeeder extends Seeder
{
    /**
     * Run the database seeds.
     */
    public function run(): void
    {
        $logs = [
            [
                'container_id' => 1,
                'location_from' => 'Pabrik Kimia - Cilegon',
                'location_to' => 'Gudang Utama - Jakarta Utara',
                'status_change' => 'In Transit → Active',
                'notes' => 'Kontainer diterima di gudang utama. Kondisi baik.',
                'logged_at' => '2024-11-10 08:30:00',
            ],
            [
                'container_id' => 1,
                'location_from' => 'Gudang Utama - Jakarta Utara',
                'location_to' => 'Gudang Utama - Jakarta Utara',
                'status_change' => 'Active → Active',
                'notes' => 'Pengisian limbah cair batch #45. Level naik 500L.',
                'logged_at' => '2024-11-20 14:15:00',
            ],
            [
                'container_id' => 2,
                'location_from' => 'Gudang Utama - Jakarta Utara',
                'location_to' => 'Zona B - Tangerang',
                'status_change' => 'Active → Full',
                'notes' => 'Kontainer penuh, dipindahkan ke zona B untuk pengangkutan.',
                'logged_at' => '2024-10-25 10:00:00',
            ],
            [
                'container_id' => 3,
                'location_from' => 'Laboratorium R&D - Bogor',
                'location_to' => 'Area Kimia - Bekasi',
                'status_change' => 'In Transit → Active',
                'notes' => 'Kontainer gas tiba dari lab. Segel masih utuh.',
                'logged_at' => '2024-12-01 09:45:00',
            ],
            [
                'container_id' => 4,
                'location_from' => 'Zona B - Tangerang',
                'location_to' => 'Depo Sementara - Karawang',
                'status_change' => 'Active → Maintenance',
                'notes' => 'Kontainer dipindahkan untuk perbaikan kebocoran kecil.',
                'logged_at' => '2024-12-10 16:30:00',
            ],
            [
                'container_id' => 5,
                'location_from' => 'Pabrik Tekstil - Semarang',
                'location_to' => 'Gudang Utama - Jakarta Utara',
                'status_change' => 'In Transit → Active',
                'notes' => 'Limbah padat dari pabrik tekstil. Berat terukur: 4500kg.',
                'logged_at' => '2024-10-01 07:00:00',
            ],
            [
                'container_id' => 8,
                'location_from' => 'Gudang Utama - Jakarta Utara',
                'location_to' => 'Depo Sementara - Karawang',
                'status_change' => 'Active → Archived',
                'notes' => 'Kontainer sudah tidak layak pakai. Di-archive.',
                'logged_at' => '2024-08-20 11:00:00',
            ],
        ];

        foreach ($logs as $log) {
            TrackingLog::create($log);
        }
    }
}
