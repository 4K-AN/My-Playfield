// Konfigurasi dasar Axios
const api = axios.create({
    baseURL: 'http://127.0.0.1:8000/api',
    headers: { 'Content-Type': 'application/json' }
});

const outputList = document.getElementById('output-list');
const errorDiv = document.getElementById('error-message');
const successDiv = document.getElementById('success-message');

// Fungsi untuk membersihkan error sebelumnya
function clearError() { 
    errorDiv.innerHTML = '';
    errorDiv.classList.remove('show');
}

// Fungsi untuk membersihkan success message sebelumnya
function clearSuccess() {
    successDiv.innerHTML = '';
    successDiv.classList.remove('show');
}

// Fungsi untuk menampilkan success message
function showSuccess(message) {
    successDiv.innerHTML = `<strong>Berhasil:</strong> ${message}`;
    successDiv.classList.add('show');
    setTimeout(() => {
        clearSuccess();
    }, 3000);
}

// Fungsi untuk merender data JSON menjadi list elemen HTML
function renderList(students) {
    outputList.innerHTML = '';
    if (students.length === 0) {
        outputList.innerHTML = '<p style="color: #999; text-align: center;">Belum ada data mahasiswa.</p>';
        return;
    }
    
    students.forEach(std => {
        let mkHtml = '<ul>';
        if(std.mataKuliah && std.mataKuliah.length > 0) {
            std.mataKuliah.forEach(mk => {
                mkHtml += `<li>${mk.kode} - ${mk.nama} (${mk.sks} SKS)</li>`;
            });
        } else {
            mkHtml += '<li style="color: #999;">Tidak ada mata kuliah</li>';
        }
        mkHtml += '</ul>';

        outputList.innerHTML += `
            <div class="student-card">
                <strong>NIM:</strong> ${std.nim} <br>
                <strong>Nama:</strong> ${std.nama} <br>
                <strong>Mata Kuliah yang diambil:</strong> ${mkHtml}
            </div>
        `;
    });
}

// Fungsi untuk menangkap dan menampilkan error validasi ke layar
function showError(error) {
    clearSuccess();
    if (error.response && error.response.data) {
        // Menggabungkan pesan error dari backend Laravel
        if (error.response.data.errors) {
            let errMessages = Object.values(error.response.data.errors).flat().join('<br>');
            errorDiv.innerHTML = `<strong>Error Validasi:</strong><br>${errMessages}`;
        } else if (error.response.data.message) {
            errorDiv.innerHTML = `<strong>Error:</strong> ${error.response.data.message}`;
        } else {
            errorDiv.innerHTML = `<strong>Error:</strong> ${JSON.stringify(error.response.data)}`;
        }
    } else if (error.message) {
        errorDiv.innerHTML = `<strong>Error:</strong> ${error.message}`;
    } else {
        errorDiv.innerHTML = `<strong>Error:</strong> Terjadi kesalahan yang tidak diketahui`;
    }
    errorDiv.classList.add('show');
}

// --- FUNGSI CRUD MENGGUNAKAN AXIOS ---

// 1. READ (Ambil semua data)
function getStudents() {
    clearError();
    clearSuccess();
    api.get('/students')
        .then(response => {
            renderList(response.data);
            showSuccess('Data mahasiswa berhasil dimuat');
        })
        .catch(error => {
            console.error('Error:', error);
            showError(error);
        });
}

// 2. CREATE (Simpan data baru)
function saveStudent() {
    clearError();
    clearSuccess();
    const nim = document.getElementById('nim').value;
    const nama = document.getElementById('nama').value;
    const kode = document.getElementById('kode').value;
    const nama_mk = document.getElementById('nama_mk').value;
    const sks = document.getElementById('sks').value;

    // Validasi input di frontend
    if (!nim || !nama || !kode || !nama_mk || !sks) {
        showError({
            response: {
                data: {
                    errors: {
                        'form': ['Semua field harus diisi!']
                    }
                }
            }
        });
        return;
    }

    const data = {
        nim: nim,
        nama: nama,
        mataKuliah: [{
            kode: kode,
            nama: nama_mk,
            sks: parseInt(sks) || 0
        }]
    };

    api.post('/students', data)
        .then(response => {
            showSuccess('Mahasiswa berhasil disimpan!');
            // Clear form
            document.getElementById('nim').value = '';
            document.getElementById('nama').value = '';
            document.getElementById('kode').value = '';
            document.getElementById('nama_mk').value = '';
            document.getElementById('sks').value = '';
            // Refresh tabel setelah simpan
            getStudents();
        })
        .catch(error => {
            console.error('Error:', error);
            showError(error);
        });
}

// 3. UPDATE PATCH (Ubah nama berdasarkan NIM)
function updateStudentName() {
    clearError();
    clearSuccess();
    const nim = document.getElementById('nim').value;
    const namaBaru = document.getElementById('nama').value;

    if(!nim) {
        showError({
            response: {
                data: {
                    errors: {
                        'nim': ['NIM harus diisi sebagai target perubahan!']
                    }
                }
            }
        });
        return;
    }

    if(!namaBaru) {
        showError({
            response: {
                data: {
                    errors: {
                        'nama': ['Nama baru harus diisi!']
                    }
                }
            }
        });
        return;
    }

    api.patch(`/students/${nim}`, { nama: namaBaru })
        .then(response => {
            showSuccess('Nama mahasiswa berhasil diperbarui!');
            getStudents();
        })
        .catch(error => {
            console.error('Error:', error);
            showError(error);
        });
}

// 4. DELETE (Hapus data berdasarkan NIM)
function deleteStudent() {
    clearError();
    clearSuccess();
    const nim = document.getElementById('nim').value;

    if(!nim) {
        showError({
            response: {
                data: {
                    errors: {
                        'nim': ['NIM harus diisi untuk menghapus data!']
                    }
                }
            }
        });
        return;
    }

    if (!confirm(`Apakah Anda yakin ingin menghapus mahasiswa dengan NIM ${nim}?`)) {
        return;
    }

    api.delete(`/students/${nim}`)
        .then(response => {
            showSuccess('Data mahasiswa berhasil dihapus!');
            document.getElementById('nim').value = '';
            getStudents();
        })
        .catch(error => {
            console.error('Error:', error);
            showError(error);
        });
}
