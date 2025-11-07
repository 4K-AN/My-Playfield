// proses.php
<?php
// Menggunakan $_GET karena metodenya get
if (isset($_GET['nama'])) {
    $nama = $_GET['nama'];
    echo "Nama yang dikirim: <b>$nama</b>";
}
?>