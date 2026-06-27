<?php

echo "<pre>";

$daftarBarang = [
    "Buku Tulis" => 15000,
    "Pensil 2B" => 5000,
    "Penghapus" => 3000,
    "Penggaris 30cm" => 7000
];
$totalBelanja = 0;
$totalBayar = 0;
$diskon = 0;

echo "==============================<br>";

echo "<h4>Selamat Datang di Toko ATK </h4>";
echo "Daftar Barang Tersedia:<br>";
echo "==============================<br>";
foreach ($daftarBarang as $nama => $harga) {
   
    echo "- " . $nama . ": Rp " . number_format($harga, 0, ',', '.') . "<br>";
}
echo "==============================<br><br>"; 

$keranjangBelanja = [
    "Buku Tulis" => 5,
    "Pensil 2B" => 5,
    "Penghapus" => 2
];



$keranjangBelanja = [
    "Buku Tulis" => 5,
    "Pensil 2B" => 5,
    "Penghapus" => 2
];

function HitungTotal(array $keranjang, array $stokBarang): int
{
    $total = 0;
    foreach ($keranjang as $namaBarang => $jumlah) {
        if (isset($stokBarang[$namaBarang])) {
            $total += $stokBarang[$namaBarang] * $jumlah;
        }
    }
    return $total;
}

$totalBelanja = HitungTotal($keranjangBelanja, $daftarBarang);
echo "Total Belanja Anda: Rp " . number_format($totalBelanja, 0, ',', '.') . "<br>";
if ($totalBelanja > 100000) {
    $diskon = $totalBelanja * 0.1;
    echo "Selamat! Anda mendapatkan diskon 10%: Rp " . number_format($diskon, 0, ',', '.') . "<br>";
    $totalBayar = $totalBelanja - $diskon;
} else {
    echo "Maaf, Anda tidak mendapatkan diskon.<br>";
    $totalBayar = $totalBelanja;
}

echo "------------------------------<br>";
echo "Total yang Harus Dibayar: Rp " . number_format($totalBayar, 0, ',', '.') . "<br>";
echo "------------------------------<br>";
echo "Terima kasih telah berbelanja di Toko ATK!<br>";
echo "==============================<br>";


echo "</pre>";

?>