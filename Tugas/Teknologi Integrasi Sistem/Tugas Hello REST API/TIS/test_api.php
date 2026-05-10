<?php

require 'vendor/autoload.php';

$client = new \GuzzleHttp\Client(['base_uri' => 'http://127.0.0.1:8000']);

function login($client, $email, $password) {
    $response = $client->post('/api/login', [
        'json' => [
            'email' => $email,
            'password' => $password
        ],
        'http_errors' => false,
        'headers' => ['Accept' => 'application/json']
    ]);
    return json_decode($response->getBody(), true)['token'] ?? null;
}

$userToken = login($client, 'user@example.com', 'password123');
$adminToken = login($client, 'admin@example.com', 'secret321');
$managerToken = login($client, 'manager@example.com', 'manager123');

function testEndpoint($client, $name, $endpoint, $token) {
    echo "Test: $name\n";
    $response = $client->get($endpoint, [
        'headers' => [
            'Authorization' => "Bearer $token",
            'Accept' => 'application/json'
        ],
        'http_errors' => false
    ]);
    echo "Status: " . $response->getStatusCode() . "\n";
    echo "Response: " . $response->getBody() . "\n";
    echo "-------------------------\n";
}

testEndpoint($client, 'User -> Admin Dashboard', '/api/admin/dashboard', $userToken);
testEndpoint($client, 'User -> User Dashboard', '/api/user/dashboard', $userToken);
testEndpoint($client, 'Admin -> Admin Dashboard', '/api/admin/dashboard', $adminToken);
testEndpoint($client, 'Admin -> User Dashboard', '/api/user/dashboard', $adminToken);
testEndpoint($client, 'Manager -> Manager Dashboard', '/api/manager/dashboard', $managerToken);
testEndpoint($client, 'Manager -> Admin Dashboard', '/api/admin/dashboard', $managerToken);
