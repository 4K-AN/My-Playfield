const api = axios.create({
    baseURL: 'http://localhost:8000/api',
    headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
});

const elements = {
    form: document.getElementById('container-form'),
    list: document.getElementById('container-list'),
    totalWeight: document.getElementById('total-weight'),
    btnSearch: document.getElementById('btn-search'),
    filterType: document.getElementById('filter-type'),
    filterWeight: document.getElementById('filter-weight'),
    modal: document.getElementById('logs-modal'),
    modalContent: document.getElementById('logs-content'),
    closeModal: document.querySelector('.close-modal')
};

const clearErrors = () => {
    document.querySelectorAll('.error-message').forEach(el => el.textContent = '');
};

const showError = (field, message) => {
    const errorEl = document.getElementById(`error-${field}`);
    if (errorEl) {
        errorEl.textContent = message;
    }
};

const calculateTotal = (data) => {
    const total = data.reduce((sum, item) => sum + parseFloat(item.weight_kg), 0);
    elements.totalWeight.textContent = total.toLocaleString('id-ID');
};

const renderContainers = (data) => {
    elements.list.innerHTML = '';
    
    if (data.length === 0) {
        elements.list.innerHTML = '<p>Tidak ada data kontainer.</p>';
        return;
    }

    data.forEach(item => {
        const card = document.createElement('div');
        card.className = 'card';
        card.innerHTML = `
            <div class="card-header">
                <span class="card-id">${item.container_id}</span>
                <span class="status-badge ${item.status === 'Active' ? 'status-active' : 'status-archived'}">${item.status}</span>
            </div>
            <div class="card-body">
                <p><strong>Tipe:</strong> ${item.waste_type}</p>
                <p><strong>Berat:</strong> ${item.weight_kg} Kg</p>
            </div>
            <div class="card-footer">
                ${item.status === 'Active' ? `<button class="btn-action btn-archive" onclick="archiveContainer('${item.container_id}')">Archive</button>` : ''}
                <button class="btn-action btn-delete" onclick="deleteContainer('${item.container_id}')">Hapus</button>
                <button class="btn-action btn-logs" onclick="showLogs('${item.container_id}')">Lihat Log</button>
            </div>
        `;
        elements.list.appendChild(card);
    });

    calculateTotal(data);
};

const fetchContainers = async () => {
    try {
        const response = await api.get('/containers');
        renderContainers(response.data);
    } catch (error) {
        alert('Gagal mengambil data kontainer.');
    }
};

const searchContainers = async () => {
    const type = elements.filterType.value;
    const minWeight = elements.filterWeight.value;
    
    let url = '/containers/search?';
    if (type) url += `type=${encodeURIComponent(type)}&`;
    if (minWeight) url += `min_weight=${encodeURIComponent(minWeight)}&`;

    try {
        const response = await api.get(url);
        renderContainers(response.data);
    } catch (error) {
        alert('Gagal melakukan pencarian.');
    }
};

elements.form.addEventListener('submit', async (e) => {
    e.preventDefault();
    clearErrors();

    const formData = new FormData(elements.form);
    const data = {
        container_id: formData.get('container_id'),
        waste_type: formData.get('waste_type'),
        weight_kg: formData.get('weight_kg')
    };

    try {
        await api.post('/containers', data);
        elements.form.reset();
        fetchContainers();
        alert('Kontainer berhasil ditambahkan.');
    } catch (error) {
        if (error.response && error.response.status === 422) {
            const errors = error.response.data.errors;
            for (const key in errors) {
                showError(key, errors[key][0]);
            }
        } else {
            alert('Terjadi kesalahan saat menyimpan data.');
        }
    }
});

elements.btnSearch.addEventListener('click', searchContainers);

window.archiveContainer = async (id) => {
    try {
        await api.patch(`/containers/${id}/archive`);
        fetchContainers();
    } catch (error) {
        alert('Gagal mengarsipkan kontainer.');
    }
};

window.deleteContainer = async (id) => {
    if (!confirm('Apakah Yakin Ingin Menghapus Kontainer Ini?')) {
        return;
    }

    try {
        await api.delete(`/containers/${id}`);
        fetchContainers();
    } catch (error) {
        alert('Gagal menghapus kontainer.');
    }
};

window.showLogs = async (id) => {
    try {
        const response = await api.get(`/containers/${id}/logs`);
        const logs = response.data;
        
        elements.modalContent.innerHTML = '';
        if (logs.length === 0) {
            elements.modalContent.innerHTML = '<p>Tidak ada log.</p>';
        } else {
            logs.forEach(log => {
                const logItem = document.createElement('div');
                logItem.className = 'log-item';
                
                const timeStr = new Date(log.timestamp).toLocaleString('id-ID');
                
                logItem.innerHTML = `
                    <div class="log-time">${timeStr}</div>
                    <div class="log-desc">${log.description}</div>
                    <div class="log-loc">${log.location}</div>
                `;
                elements.modalContent.appendChild(logItem);
            });
        }
        
        elements.modal.classList.add('active');
    } catch (error) {
        alert('Gagal mengambil log kontainer.');
    }
};

elements.closeModal.addEventListener('click', () => {
    elements.modal.classList.remove('active');
});

window.addEventListener('click', (e) => {
    if (e.target === elements.modal) {
        elements.modal.classList.remove('active');
    }
});

document.addEventListener('DOMContentLoaded', fetchContainers);
