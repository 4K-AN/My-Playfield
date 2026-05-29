<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        Schema::create('containers', function (Blueprint $table) {
            $table->id();
            $table->string('container_code')->unique();
            $table->enum('type', ['limbah_cair', 'limbah_padat', 'limbah_gas']);
            $table->decimal('capacity', 10, 2);
            $table->decimal('current_fill_level', 10, 2)->default(0);
            $table->string('location');
            $table->enum('status', ['Active', 'Maintenance', 'Archived', 'Full'])->default('Active');
            $table->date('last_maintenance_date')->nullable();
            $table->timestamps();
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('containers');
    }
};
