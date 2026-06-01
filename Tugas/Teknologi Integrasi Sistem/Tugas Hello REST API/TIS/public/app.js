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
    
    // Validasi: Semua field harus diisi
    if (!nim || !nama || !kode || !nama_mk || !sks) {
        const msg = 'Semua field harus diisi!';
        errorDiv.innerHTML = 'Error Validasi: ' + msg;
        errorDiv.classList.add('show');
        return;
    }

    // Validasi: NIM harus 15 digit
    if (nim.length !== 15 || isNaN(nim)) {
        const msg = 'NIM harus terdiri dari 15 digit angka!';
        errorDiv.innerHTML = 'Error Validasi: ' + msg;
        errorDiv.classList.add('show');
        return;
    }

    // Validasi: Nama minimal 3 karakter (sesuai backend)
    if (nama.length < 3) {
        const msg = 'Nama harus minimal 3 karakter!';
        errorDiv.innerHTML = 'Error Validasi: ' + msg;
        errorDiv.classList.add('show');
        return;
    }

    // Validasi: Nama maksimal 50 karakter
    if (nama.length > 50) {
        const msg = 'Nama maksimal 50 karakter!';
        errorDiv.innerHTML = 'Error Validasi: ' + msg;
        errorDiv.classList.add('show');
        return;
    }

    // Validasi: Format kode mata kuliah (3 huruf + 5 angka)
    const kodeRegex = /^[A-Z]{3}[0-9]{5}$/;
    if (!kodeRegex.test(kode)) {
        const msg = 'Kode mata kuliah harus format 3 huruf kapital + 5 angka (contoh: CIE61205)!';
        errorDiv.innerHTML = 'Error Validasi: ' + msg;
        errorDiv.classList.add('show');
        return;
    }

    // Validasi: SKS antara 1-6
    const sksNum = parseInt(sks);
    if (isNaN(sksNum) || sksNum < 1 || sksNum > 6) {
        const msg = 'SKS harus antara 1-6!';
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
            sks: sksNum
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

    // Validasi: NIM harus 15 digit
    if (nim.length !== 15 || isNaN(nim)) {
        const msg = 'NIM harus terdiri dari 15 digit angka!';
        errorDiv.innerHTML = 'Error Validasi: ' + msg;
        errorDiv.classList.add('show');
        return;
    }
    
    if(!namaBaru) {
        const msg = 'Harap isi kolom Nama dengan nama baru!';
        errorDiv.innerHTML = 'Error: ' + msg;
        errorDiv.classList.add('show');
        return;
    }

    // Validasi: Nama minimal 3 karakter (sesuai backend)
    if (namaBaru.length < 3) {
        const msg = 'Nama baru harus minimal 3 karakter!';
        errorDiv.innerHTML = 'Error Validasi: ' + msg;
        errorDiv.classList.add('show');
        return;
    }

    // Validasi: Nama maksimal 50 karakter
    if (namaBaru.length > 50) {
        const msg = 'Nama maksimal 50 karakter!';
        errorDiv.innerHTML = 'Error Validasi: ' + msg;
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

// ========== NEW FUNCTIONS FOR COMPOUND DATA & NESTED RESOURCE ==========

function viewStudentCompound() {
    clearMessages();
    
    const nim = document.getElementById('nim').value.trim();

    if(!nim) {
        const msg = 'Harap isi kolom NIM untuk melihat data!';
        errorDiv.innerHTML = 'Error: ' + msg;
        errorDiv.classList.add('show');
        return;
    }

    // Validasi: NIM harus 15 digit
    if (nim.length !== 15 || isNaN(nim)) {
        const msg = 'NIM harus terdiri dari 15 digit angka!';
        errorDiv.innerHTML = 'Error Validasi: ' + msg;
        errorDiv.classList.add('show');
        return;
    }

    outputList.innerHTML = '<div class="loading">Mengambil data mahasiswa...</div>';

    // Endpoint Compound Data: GET /api/students/{nim}
    api.get('/students/' + nim)
        .then(function(response) {
            const data = response.data.data || response.data;
            renderList([data]); // Render single student
            showSuccess('Data mahasiswa berhasil dimuat (Compound Data)');
        })
        .catch(function(error) {
            showError(error);
            outputList.innerHTML = '<div class="no-data">Gagal mengambil data mahasiswa</div>';
        });
}

function viewStudentCourses() {
    clearMessages();
    
    const nim = document.getElementById('nim').value.trim();

    if(!nim) {
        const msg = 'Harap isi kolom NIM untuk melihat mata kuliah!';
        errorDiv.innerHTML = 'Error: ' + msg;
        errorDiv.classList.add('show');
        return;
    }

    // Validasi: NIM harus 15 digit
    if (nim.length !== 15 || isNaN(nim)) {
        const msg = 'NIM harus terdiri dari 15 digit angka!';
        errorDiv.innerHTML = 'Error Validasi: ' + msg;
        errorDiv.classList.add('show');
        return;
    }

    outputList.innerHTML = '<div class="loading">Mengambil daftar mata kuliah...</div>';

    // Endpoint Nested Resource: GET /api/students/{nim}/mata-kuliah
    api.get('/students/' + nim + '/mata-kuliah')
        .then(function(response) {
            const courses = response.data.data || [];
            const studentNim = response.data.student_nim || nim;
            
            outputList.innerHTML = '';
            
            if (!courses || courses.length === 0) {
                outputList.innerHTML = '<div class="no-data">Tidak ada mata kuliah untuk NIM ' + studentNim + '</div>';
                showSuccess('Data mata kuliah berhasil dimuat (Nested Resource)');
                return;
            }
            
            let mkHtml = '<div class="student-card">';
            mkHtml += '<div><strong>NIM:</strong> ' + studentNim + '</div>';
            mkHtml += '<div><strong>Daftar Mata Kuliah (Nested Resource):</strong><ul>';
            
            courses.forEach(function(mk) {
                mkHtml += '<li><strong>' + mk.kode + '</strong> - ' + mk.nama + ' (' + mk.sks + ' SKS)</li>';
            });
            
            mkHtml += '</ul></div></div>';
            
            outputList.innerHTML = mkHtml;
            showSuccess('Data mata kuliah berhasil dimuat (Nested Resource)');
        })
        .catch(function(error) {
            showError(error);
            outputList.innerHTML = '<div class="no-data">Gagal mengambil daftar mata kuliah</div>';
        });
}

document.addEventListener('DOMContentLoaded', function() {
    getStudents();
});
