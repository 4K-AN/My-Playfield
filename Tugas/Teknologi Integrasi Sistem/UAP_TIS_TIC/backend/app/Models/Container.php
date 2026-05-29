<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\HasMany;

class Container extends Model
{
    use HasFactory;

    /**
     * The attributes that are mass assignable.
     *
     * @var list<string>
     */
    protected $fillable = [
        'container_code',
        'type',
        'capacity',
        'current_fill_level',
        'location',
        'status',
        'last_maintenance_date',
    ];

    /**
     * Get the attributes that should be cast.
     *
     * @return array<string, string>
     */
    protected function casts(): array
    {
        return [
            'capacity' => 'decimal:2',
            'current_fill_level' => 'decimal:2',
            'last_maintenance_date' => 'date',
        ];
    }

    /**
     * Get the tracking logs for the container.
     */
    public function trackingLogs(): HasMany
    {
        return $this->hasMany(TrackingLog::class);
    }
}
