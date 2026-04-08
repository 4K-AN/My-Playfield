// Konfigurasi dasar Axios
const api = axios.create({
    baseURL: 'http://127.0.0.1:8000/api',
    headers: { 'Content-Type': 'application/json' }
});

const outputList = document.getElementById('output-list');
const errorDiv = document.getElementById('error-message');
const successDiv = document.getElementById('success-message');

// Fungsi untuk membersihkan error/success sebelumnya
function clearMessages() {
    errorDiv.innerHTML = '';
    errorDiv.classList.remove('show');
    successDiv.innerHTML = '';
    successDiv.classList.remove('show');
}

// Fungsi untuk menampilkan pesan sukses
function showSuccess(message) {
    clearMessages();
    successDiv.innerHTML = `<strong>✓ Sukses:</strong> ${message}`;
    successDiv.classList.add('show');
    setTimeout(() => {
        successDiv.classList.remove('show');
    }, 5000);
}

// Fungsi untuk merender data JSON menjadi list elemen HTML
function renderList(students) {
    outputList.innerHTML = '';
    
    if (!students || students.length === 0) {
        outputList.innerHTML = '<div class="no-data">📭 Belum ada data mahasiswa.</div>';
        return;
    }
    
    students.forEach((std, index) => {
        let mkHtml = '<ul>';
        if(std.mataKuliah && std.mataKuliah.length > 0) {
            std.mataKuliah.forEach(mk => {
                mkHtml += `<li><strong>${mk.kode}</strong> - ${mk.nama} (${mk.sks} SKS)</li>`;
            });
        } else {
            mkHtml += '<li><em>Tidak ada mata kuliah</em></li>';
        }
        mkHtml += '</ul>';

        outputList.innerHTML += `
            <div class="student-card">
                <div><strong>NIM:</strong> ${std.nim}</div>
                <div><strong>Nama:</strong> ${std.nama}</div>
                <div><strong>Mata Kuliah yang diambil:</strong> ${mkHtml}</div>
            </div>
        `;
    });
}

// Fungsi untuk menangkap dan menampilkan error validasi ke layar
function showError(error) {
    clearMessages();
    
    if (error.response && error.response.data) {
        // Jika ada response dari server
        if (error.response.data.errors) {
            // Menggabungkan pesan error dari backend Laravel
            let errMessages = Object.entries(error.response.data.errors)
                .map(([field, messages]) => {
                    const msgs = Array.isArray(messages) ? messages : [messages];
                    return `<strong>${field}:</strong> ${msgs.join(', ')}`;
                })
                .join('<br>');
            errorDiv.innerHTML = `<strong>❌ Error Validasi:</strong><br>${errMessages}`;
        } else if (error.response.data.message) {
            errorDiv.innerHTML = `<strong>❌ Error:</strong> ${error.response.data.message}`;
        } else {
            errorDiv.innerHTML = `<strong>❌ Error:</strong> ${JSON.stringify(error.response.data)}`;
        }
    } else if (error.message) {
        errorDiv.innerHTML = `<strong>❌ Error:</strong> ${error.message}`;
    } else {
        errorDiv.innerHTML = `<strong>❌ Error:</strong> Terjadi kesalahan yang tidak terduga`;
    }
    
    errorDiv.classList.add('show');
}

// --- FUNGSI CRUD MENGGUNAKAN AXIOS ---

// 1. READ (Ambil semua data)
function getStudents() {
    clearMessages();
    outputList.innerHTML = '<div class="loading">⏳ Mengambil data...</div>';
    
    api.get('/students')
        .then(response => {
            renderList(response.data);
            showSuccess('Data mahasiswa berhasil dimuat');
        })
        .catch(error => {
            showError(error);
            outputList.innerHTML = '<div class="no-data">📭 Gagal mengambil data</div>';
        });
}

// 2. CREATE (Simpan data baru)
function saveStudent() {
    clearMessages();
    
    const nim = document.getElementById('nim').value.trim();
    const nama = document.getElementById('nama').value.trim();
    const kode = document.getElementById('kode').value.trim();
    const nama_mk = document.getElementById('nama_mk').value.trim();
    const sks = document.getElementById('sks').value;
    
    // Validasi di client-side
    if (!nim || !nama || !kode || !nama_mk || !sks) {
        errorDiv.innerHTML = `<strong>❌ Error Validasi:</strong><br>
            Semua field harus diisi!`;
        errorDiv.classList.add('show');
        return;
    }
    
    const data = {
        nim: nim,
        nama: nama,
        mataKuliah: [{
            kode: kode,
            nama: nama_mk,
            sks: parseInt(sks)
        }]
    };

    outputList.innerHTML = '<div class="loading">⏳ Menyimpan data...</div>';

    api.post('/students', data)
        .then(response => {
            showSuccess('Mahasiswa berhasil disimpan!');
            // Clear form
            document.getElementById('nim').value = '';
            document.getElementById('nama').value = '';
            document.getElementById('kode').value = '';
            document.getElementById('nama_mk').value = '';
            document.getElementById('sks').value = '';
            // Refresh data
            setTimeout(() => getStudents(), 500);
        })
        .catch(error => showError(error));
}

// 3. UPDATE PATCH (Ubah nama berdasarkan NIM)
function updateStudentName() {
    clearMessages();
    
    const nim = document.getElementById('nim').value.trim();
    const namaBaru = document.getElementById('nama').value.trim();

    if(!nim) {
        errorDiv.innerHTML = `<strong>❌ Error:</strong><br>
            Harap isi kolom NIM sebagai target perubahan!`;
        errorDiv.classList.add('show');
        return;
    }
    
    if(!namaBaru) {
        errorDiv.innerHTML = `<strong>❌ Error:</strong><br>
            Harap isi kolom Nama dengan nama baru!`;
        errorDiv.classList.add('show');
        return;
    }

    outputList.innerHTML = '<div class="loading">⏳ Mengubah nama...</div>';

    api.patch(`/students/${nim}`, { nama: namaBaru })
        .then(response => {
            showSuccess(`Nama mahasiswa (NIM: ${nim}) berhasil diperbarui!`);
            document.getElementById('nama').value = '';
            setTimeout(() => getStudents(), 500);
        })
        .catch(error => showError(error));
}

// 4. DELETE (Hapus data berdasarkan NIM)
function deleteStudent() {
    clearMessages();
    
    const nim = document.getElementById('nim').value.trim();

    if(!nim) {
        errorDiv.innerHTML = `<strong>❌ Error:</strong><br>
            Harap isi kolom NIM yang akan dihapus!`;
        errorDiv.classList.add('show');
        return;
    }

    if(!confirm(`Apakah Anda yakin ingin menghapus mahasiswa dengan NIM: ${nim}?`)) {
        return;
    }

    outputList.innerHTML = '<div class="loading">⏳ Menghapus data...</div>';

    api.delete(`/students/${nim}`)
        .then(response => {
            showSuccess(`Mahasiswa dengan NIM ${nim} berhasil dihapus!`);
            document.getElementById('nim').value = '';
            setTimeout(() => getStudents(), 500);
        })
        .catch(error => showError(error));
}

// Load data saat halaman pertama kali dibuka
document.addEventListener('DOMContentLoaded', function() {
    getStudents();
});
