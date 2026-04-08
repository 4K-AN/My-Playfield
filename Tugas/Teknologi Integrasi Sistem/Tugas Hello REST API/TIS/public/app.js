const api = axios.create({
    baseURL: 'http://127.0.0.1:8000/api',
    headers: { 'Content-Type': 'application/json' }
});

const outputList = document.getElementById('output-list');
const errorDiv = document.getElementById('error-message');
const successDiv = document.getElementById('success-message');

function clearMessages() {
    errorDiv.innerHTML = '';
    errorDiv.classList.remove('show');
    successDiv.innerHTML = '';
    successDiv.classList.remove('show');
}

function clearForm() {
    document.getElementById('nim').value = '';
    document.getElementById('nama').value = '';
    document.getElementById('kode').value = '';
    document.getElementById('nama_mk').value = '';
    document.getElementById('sks').value = '';
}

function showSuccess(message) {
    clearMessages();
    successDiv.innerHTML = '<strong>Sukses:</strong> ' + message;
    successDiv.classList.add('show');
    setTimeout(function() {
        successDiv.classList.remove('show');
    }, 5000);
}

function renderList(students) {
    outputList.innerHTML = '';
    
    if (!students || students.length === 0) {
        outputList.innerHTML = '<div class="no-data">Belum ada data mahasiswa.</div>';
        return;
    }
    
    students.forEach(function(std, index) {
        let mkHtml = '<ul>';
        if(std.mataKuliah && std.mataKuliah.length > 0) {
            std.mataKuliah.forEach(function(mk) {
                mkHtml += '<li><strong>' + mk.kode + '</strong> - ' + mk.nama + ' (' + mk.sks + ' SKS)</li>';
            });
        } else {
            mkHtml += '<li><em>Tidak ada mata kuliah</em></li>';
        }
        mkHtml += '</ul>';

        outputList.innerHTML += '<div class="student-card">' +
            '<div><strong>NIM:</strong> ' + std.nim + '</div>' +
            '<div><strong>Nama:</strong> ' + std.nama + '</div>' +
            '<div><strong>Mata Kuliah yang diambil:</strong> ' + mkHtml + '</div>' +
            '</div>';
    });
}

function showError(error) {
    clearMessages();
    
    if (error.response && error.response.data) {
        if (error.response.data.errors) {
            let errMessages = '';
            for (let field in error.response.data.errors) {
                let msgs = error.response.data.errors[field];
                if (!Array.isArray(msgs)) {
                    msgs = [msgs];
                }
                errMessages += '<strong>' + field + ':</strong> ' + msgs.join(', ') + '<br>';
            }
            errorDiv.innerHTML = '<strong>Error Validasi:</strong><br>' + errMessages;
        } else if (error.response.data.message) {
            errorDiv.innerHTML = '<strong>Error:</strong> ' + error.response.data.message;
        } else {
            errorDiv.innerHTML = '<strong>Error:</strong> ' + JSON.stringify(error.response.data);
        }
    } else if (error.message) {
        errorDiv.innerHTML = '<strong>Error:</strong> ' + error.message;
    } else {
        errorDiv.innerHTML = '<strong>Error:</strong> Terjadi kesalahan yang tidak terduga';
    }
    
    errorDiv.classList.add('show');
}

function getStudents() {
    clearMessages();
    outputList.innerHTML = '<div class="loading">Mengambil data...</div>';
    
    api.get('/students')
        .then(function(response) {
            renderList(response.data);
            showSuccess('Data mahasiswa berhasil dimuat');
        })
        .catch(function(error) {
            showError(error);
            outputList.innerHTML = '<div class="no-data">Gagal mengambil data</div>';
        });
}

function saveStudent() {
    clearMessages();
    
    const nim = document.getElementById('nim').value.trim();
    const nama = document.getElementById('nama').value.trim();
    const kode = document.getElementById('kode').value.trim();
    const nama_mk = document.getElementById('nama_mk').value.trim();
    const sks = document.getElementById('sks').value;
    
    if (!nim || !nama || !kode || !nama_mk || !sks) {
        const msg = 'Semua field harus diisi!';
        errorDiv.innerHTML = 'Error Validasi: ' + msg;
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

    outputList.innerHTML = '<div class="loading">Menyimpan data...</div>';

    api.post('/students', data)
        .then(function(response) {
            clearForm();
            showSuccess('Mahasiswa berhasil disimpan!');
            setTimeout(function() {
                getStudents();
            }, 800);
        })
        .catch(function(error) {
            showError(error);
        });
}

function updateStudentName() {
    clearMessages();
    
    const nim = document.getElementById('nim').value.trim();
    const namaBaru = document.getElementById('nama').value.trim();

    if(!nim) {
        const msg = 'Harap isi kolom NIM sebagai target perubahan!';
        errorDiv.innerHTML = 'Error: ' + msg;
        errorDiv.classList.add('show');
        return;
    }
    
    if(!namaBaru) {
        const msg = 'Harap isi kolom Nama dengan nama baru!';
        errorDiv.innerHTML = 'Error: ' + msg;
        errorDiv.classList.add('show');
        return;
    }

    outputList.innerHTML = '<div class="loading">Mengubah nama...</div>';

    api.patch('/students/' + nim, { nama: namaBaru })
        .then(function(response) {
            clearForm();
            showSuccess('Nama mahasiswa (NIM: ' + nim + ') berhasil diperbarui!');
            setTimeout(function() {
                getStudents();
            }, 800);
        })
        .catch(function(error) {
            showError(error);
        });
}

function deleteStudent() {
    clearMessages();
    
    const nim = document.getElementById('nim').value.trim();

    if(!nim) {
        const msg = 'Harap isi kolom NIM yang akan dihapus!';
        errorDiv.innerHTML = 'Error: ' + msg;
        errorDiv.classList.add('show');
        return;
    }

    if(!confirm('Apakah Anda yakin ingin menghapus mahasiswa dengan NIM: ' + nim + '?')) {
        return;
    }

    outputList.innerHTML = '<div class="loading">Menghapus data...</div>';

    api.delete('/students/' + nim)
        .then(function(response) {
            clearForm();
            showSuccess('Mahasiswa dengan NIM ' + nim + ' berhasil dihapus!');
            setTimeout(function() {
                getStudents();
            }, 800);
        })
        .catch(function(error) {
            showError(error);
        });
}

document.addEventListener('DOMContentLoaded', function() {
    getStudents();
});
