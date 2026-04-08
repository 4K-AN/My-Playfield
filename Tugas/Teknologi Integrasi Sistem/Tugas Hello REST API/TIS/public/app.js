// Konfigurasi dasar Axios
const api = axios.create({
    baseURL: 'http://127.0.0.1:8000/api',
    headers: { 'Content-Type': 'application/json' }
});

const outputList = document.getElementById('output-list');
const errorDiv = document.getElementById('error-message');
const successDiv = document.getElementById('success-message');

console.log('✅ app.js loaded successfully');

// Fungsi untuk membersihkan error/success sebelumnya
function clearMessages() {
    errorDiv.innerHTML = '';
    errorDiv.classList.remove('show');
    successDiv.innerHTML = '';
    successDiv.classList.remove('show');
}

// Fungsi untuk reset form
function clearForm() {
    console.log('🧹 Clearing form');
    document.getElementById('nim').value = '';
    document.getElementById('nama').value = '';
    document.getElementById('kode').value = '';
    document.getElementById('nama_mk').value = '';
    document.getElementById('sks').value = '';
}

// Fungsi untuk menampilkan pesan sukses
function showSuccess(message) {
    clearMessages();
    successDiv.innerHTML = `<strong>✓ Sukses:</strong> ${message}`;
    successDiv.classList.add('show');
    console.log('✅ ' + message);
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
    
    console.log('📋 Rendering ' + students.length + ' students');
    
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
    
    console.error('❌ Error occurred:', error);
    
    if (error.response && error.response.data) {
        console.error('Response data:', error.response.data);
        
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
    
    console.log('🔍 Fetching students...');
    api.get('/students')
        .then(response => {
            console.log('✅ Fetched successfully:', response.data);
            renderList(response.data);
            showSuccess('Data mahasiswa berhasil dimuat');
        })
        .catch(error => {
            console.error('❌ Fetch failed:', error);
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
    
    console.log('📝 Save attempt - NIM:', nim, 'Nama:', nama, 'Kode:', kode, 'Nama MK:', nama_mk, 'SKS:', sks);
    
    // Validasi di client-side
    if (!nim || !nama || !kode || !nama_mk || !sks) {
        const msg = 'Semua field harus diisi!';
        console.warn('⚠️ Validation failed:', msg);
        errorDiv.innerHTML = `<strong>❌ Error Validasi:</strong><br>${msg}`;
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

    console.log('📤 Sending POST request:', data);
    outputList.innerHTML = '<div class="loading">⏳ Menyimpan data...</div>';

    api.post('/students', data)
        .then(response => {
            console.log('✅ POST Success:', response);
            clearForm();
            showSuccess('Mahasiswa berhasil disimpan!');
            // Refresh data dengan delay lebih lama
            setTimeout(() => {
                console.log('🔄 Refreshing list after create...');
                getStudents();
            }, 800);
        })
        .catch(error => {
            console.error('❌ POST Failed:', error);
            if (error.response) {
                console.error('Response status:', error.response.status);
                console.error('Response data:', error.response.data);
            }
            showError(error);
        });
}

// 3. UPDATE PATCH (Ubah nama berdasarkan NIM)
function updateStudentName() {
    clearMessages();
    
    const nim = document.getElementById('nim').value.trim();
    const namaBaru = document.getElementById('nama').value.trim();

    console.log('✏️ Update attempt - NIM:', nim, 'New Name:', namaBaru);

    if(!nim) {
        const msg = 'Harap isi kolom NIM sebagai target perubahan!';
        console.warn('⚠️ ' + msg);
        errorDiv.innerHTML = `<strong>❌ Error:</strong><br>${msg}`;
        errorDiv.classList.add('show');
        return;
    }
    
    if(!namaBaru) {
        const msg = 'Harap isi kolom Nama dengan nama baru!';
        console.warn('⚠️ ' + msg);
        errorDiv.innerHTML = `<strong>❌ Error:</strong><br>${msg}`;
        errorDiv.classList.add('show');
        return;
    }

    outputList.innerHTML = '<div class="loading">⏳ Mengubah nama...</div>';

    console.log('📤 Sending PATCH request to /students/' + nim);
    api.patch(`/students/${nim}`, { nama: namaBaru })
        .then(response => {
            console.log('✅ PATCH Success:', response);
            clearForm();
            showSuccess(`Nama mahasiswa (NIM: ${nim}) berhasil diperbarui!`);
            setTimeout(() => {
                console.log('🔄 Refreshing list after update...');
                getStudents();
            }, 800);
        })
        .catch(error => {
            console.error('❌ PATCH Failed:', error);
            showError(error);
        });
}

// 4. DELETE (Hapus data berdasarkan NIM)
function deleteStudent() {
    clearMessages();
    
    const nim = document.getElementById('nim').value.trim();

    console.log('🗑️  Delete attempt - NIM:', nim);

    if(!nim) {
        const msg = 'Harap isi kolom NIM yang akan dihapus!';
        console.warn('⚠️ ' + msg);
        errorDiv.innerHTML = `<strong>❌ Error:</strong><br>${msg}`;
        errorDiv.classList.add('show');
        return;
    }

    if(!confirm(`Apakah Anda yakin ingin menghapus mahasiswa dengan NIM: ${nim}?`)) {
        console.log('⚠️ Delete cancelled by user');
        return;
    }

    outputList.innerHTML = '<div class="loading">⏳ Menghapus data...</div>';

    console.log('📤 Sending DELETE request to /students/' + nim);
    api.delete(`/students/${nim}`)
        .then(response => {
            console.log('✅ DELETE Success:', response);
            clearForm();
            showSuccess(`Mahasiswa dengan NIM ${nim} berhasil dihapus!`);
            setTimeout(() => {
                console.log('🔄 Refreshing list after delete...');
                getStudents();
            }, 800);
        })
        .catch(error => {
            console.error('❌ DELETE Failed:', error);
            showError(error);
        });
}

// Load data saat halaman pertama kali dibuka
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 DOM Content Loaded - fetching initial data');
    getStudents();
});
