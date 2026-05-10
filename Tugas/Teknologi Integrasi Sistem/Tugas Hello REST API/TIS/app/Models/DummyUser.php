<?php

namespace App\Models;

use Tymon\JWTAuth\Contracts\JWTSubject;

class DummyUser implements JWTSubject
{
    public $id;
    public $name;
    public $email;
    public $password;
    public $role;

    public function __construct(array $attributes = [])
    {
        $this->id = $attributes['id'] ?? null;
        $this->name = $attributes['name'] ?? null;
        $this->email = $attributes['email'] ?? null;
        $this->password = $attributes['password'] ?? null;
        $this->role = $attributes['role'] ?? null;
    }

    public function getJWTIdentifier()
    {
        return $this->id;
    }

    public function getJWTCustomClaims()
    {
        return [];
    }
}
