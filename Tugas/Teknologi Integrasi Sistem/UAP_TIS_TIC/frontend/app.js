/**
 * WowoClean - Frontend Application
 * Sistem Manajemen Kontainer Limbah B3
 *
 * Menggunakan Axios untuk komunikasi dengan API Gateway V1
 * Autentikasi: JWT Bearer Token
 */

// ==================== CONFIGURATION ====================
const API_BASE_URL = '/api/v1';
const GATEWAY_URL = `${API_BASE_URL}/gateway`;

// Axios instance dengan default config
const api = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
    },
});

// Axios interceptor - sertakan token di setiap request
api.interceptors.request.use((config) => {
    const token = localStorage.getItem('jwt_token');
    if (token) {
        config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
});

// Axios interceptor - handle 401 (token expired)
api.interceptors.response.use(
    (response) => response,
    (error) => {
        if (error.response && error.response.status === 401) {
            localStorage.removeItem('jwt_token');
            localStorage.removeItem('user_data');
            showLoginPage();
            showToast('Sesi Anda telah berakhir. Silakan login kembali.', 'warning');
        }
        return Promise.reject(error);
    }
);

// ==================== STATE ====================
let currentUser = null;
let containers = [];

// ==================== INITIALIZATION ====================
document.addEventListener('DOMContentLoaded', () => {
    const token = localStorage.getItem('jwt_token');
    const userData = localStorage.getItem('user_data');

    if (token && userData) {
        currentUser = JSON.parse(userData);
        showDashboard();
    } else {
        showLoginPage();
    }
});

// ==================== AUTH FUNCTIONS ====================
async function handleLogin(event) {
    event.preventDefault();

    const email = document.getElementById('loginEmail').value;
    const password = document.getElementById('loginPassword').value;
    const loginBtn = document.getElementById('loginBtn');
    const errorDiv = document.getElementById('loginError');

    // Loading state
    loginBtn.disabled = true;
    loginBtn.innerHTML = '<span class="loading-spinner"></span> Memproses...';
    errorDiv.style.display = 'none';

    try {
        const response = await api.post('/login', { email, password });

        if (response.data.success) {
            const { token, user } = response.data.data;

            // Simpan token dan user data ke localStorage
            localStorage.setItem('jwt_token', token);
            localStorage.setItem('user_data', JSON.stringify(user));

            currentUser = user;
            showDashboard();
            showToast(`Selamat datang, ${user.name}!`, 'success');
        }
    } catch (error) {
        const message = error.response?.data?.message || 'Terjadi kesalahan. Coba lagi.';
        errorDiv.textContent = message;
        errorDiv.style.display = 'block';
    } finally {
        loginBtn.disabled = false;
        loginBtn.innerHTML = '<i class="fas fa-sign-in-alt"></i> Masuk';
    }
}

async function handleLogout() {
    try {
        await api.post('/logout');
    } catch (error) {
        // Tetap logout meskipun request gagal
    }

    localStorage.removeItem('jwt_token');
    localStorage.removeItem('user_data');
    currentUser = null;
    showLoginPage();
    showToast('Anda telah berhasil logout.', 'success');
}

// ==================== PAGE NAVIGATION ====================
function showLoginPage() {
    document.getElementById('loginPage').style.display = 'flex';
    document.getElementById('dashboardPage').style.display = 'none';
    document.getElementById('loginForm').reset();
    document.getElementById('loginError').style.display = 'none';
}

function showDashboard() {
    document.getElementById('loginPage').style.display = 'none';
    document.getElementById('dashboardPage').style.display = 'block';

    // Update user info in navbar
    document.getElementById('userName').textContent = currentUser.name;
    const roleEl = document.getElementById('userRole');
    roleEl.textContent = currentUser.role.toUpperCase();
    roleEl.className = `user-role role-${currentUser.role}`;

    // Show admin actions
    const isAdmin = currentUser.role === 'admin';
    document.getElementById('adminActions').style.display = isAdmin ? 'flex' : 'none';
    document.getElementById('actionHeader').style.display = isAdmin ? '' : 'none';

    // Load data
    loadContainers();
}

// ==================== CONTAINER CRUD ====================
async function loadContainers() {
    const search = document.getElementById('searchInput').value;
    const type = document.getElementById('filterType').value;
    const status = document.getElementById('filterStatus').value;

    const params = new URLSearchParams();
    if (search) params.append('search', search);
    if (type) params.append('type', type);
    if (status) params.append('status', status);

    try {
        const response = await api.get(`/gateway/containers?${params.toString()}`);
        containers = response.data.data;
        renderContainers(containers);
        updateStats(containers);
    } catch (error) {
        showToast('Gagal memuat data kontainer.', 'error');
    }
}

function renderContainers(data) {
    const tbody = document.getElementById('containerTableBody');
    const isAdmin = currentUser && currentUser.role === 'admin';

    if (!data || data.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="${isAdmin ? 7 : 6}">
                    <div class="empty-state">
                        <i class="fas fa-inbox"></i>
                        <h3>Tidak ada data</h3>
                        <p>Belum ada kontainer yang tersedia atau sesuai filter.</p>
                    </div>
                </td>
            </tr>
        `;
        return;
    }

    tbody.innerHTML = data.map(c => {
        const fillPercent = c.capacity > 0 ? ((c.current_fill_level / c.capacity) * 100).toFixed(1) : 0;
        const fillClass = fillPercent > 80 ? 'fill-high' : fillPercent > 50 ? 'fill-medium' : 'fill-low';

        const statusClass = {
            'Active': 'badge-active',
            'Maintenance': 'badge-maintenance',
            'Archived': 'badge-archived',
            'Full': 'badge-full',
        }[c.status] || 'badge-active';

        const typeLabel = {
            'limbah_cair': 'Cair',
            'limbah_padat': 'Padat',
            'limbah_gas': 'Gas',
        }[c.type] || c.type;

        return `
            <tr>
                <td>
                    <strong>${escapeHtml(c.container_code)}</strong>
                </td>
                <td><span class="type-badge">${typeLabel}</span></td>
                <td>${escapeHtml(c.location)}</td>
                <td>${Number(c.capacity).toLocaleString('id-ID')}</td>
                <td>
                    <div class="fill-bar">
                        <div class="fill-bar-inner ${fillClass}" style="width: ${Math.min(fillPercent, 100)}%"></div>
                    </div>
                    <div class="fill-text">${Number(c.current_fill_level).toLocaleString('id-ID')} (${fillPercent}%)</div>
                </td>
                <td><span class="badge ${statusClass}">${c.status}</span></td>
                ${isAdmin ? `
                <td>
                    <div class="action-btns">
                        <button class="action-btn" onclick="viewTrackingLogs(${c.id}, '${escapeHtml(c.container_code)}')" title="Log Perjalanan">
                            <i class="fas fa-route"></i>
                        </button>
                        <button class="action-btn" onclick="openEditModal(${c.id})" title="Edit">
                            <i class="fas fa-edit"></i>
                        </button>
                        <button class="action-btn" onclick="archiveContainer(${c.id})" title="Archive">
                            <i class="fas fa-archive"></i>
                        </button>
                        <button class="action-btn delete" onclick="deleteContainer(${c.id})" title="Hapus">
                            <i class="fas fa-trash"></i>
                        </button>
                    </div>
                </td>
                ` : ''}
            </tr>
        `;
    }).join('');
}

function updateStats(data) {
    document.getElementById('statTotal').textContent = data.length;
    document.getElementById('statActive').textContent = data.filter(c => c.status === 'Active').length;
    document.getElementById('statMaintenance').textContent = data.filter(c => c.status === 'Maintenance').length;
    document.getElementById('statFull').textContent = data.filter(c => c.status === 'Full').length;
}

// ==================== MODAL FUNCTIONS ====================
function openAddModal() {
    document.getElementById('modalTitle').textContent = 'Tambah Kontainer Baru';
    document.getElementById('containerForm').reset();
    document.getElementById('editContainerId').value = '';
    document.getElementById('formFillLevel').value = '0';
    document.getElementById('containerModal').classList.add('active');
}

function openEditModal(id) {
    const container = containers.find(c => c.id === id);
    if (!container) return;

    document.getElementById('modalTitle').textContent = 'Edit Kontainer';
    document.getElementById('editContainerId').value = id;
    document.getElementById('formCode').value = container.container_code;
    document.getElementById('formType').value = container.type;
    document.getElementById('formCapacity').value = container.capacity;
    document.getElementById('formFillLevel').value = container.current_fill_level;
    document.getElementById('formLocation').value = container.location;
    document.getElementById('formStatus').value = container.status;
    document.getElementById('formMaintDate').value = container.last_maintenance_date
        ? container.last_maintenance_date.split('T')[0]
        : '';

    document.getElementById('containerModal').classList.add('active');
}

function closeModal() {
    document.getElementById('containerModal').classList.remove('active');
}

async function submitContainer() {
    const id = document.getElementById('editContainerId').value;
    const data = {
        container_code: document.getElementById('formCode').value,
        type: document.getElementById('formType').value,
        capacity: parseFloat(document.getElementById('formCapacity').value),
        current_fill_level: parseFloat(document.getElementById('formFillLevel').value) || 0,
        location: document.getElementById('formLocation').value,
        status: document.getElementById('formStatus').value,
        last_maintenance_date: document.getElementById('formMaintDate').value || null,
    };

    try {
        if (id) {
            // Update
            await api.put(`/gateway/containers/${id}`, data);
            showToast('Kontainer berhasil diperbarui!', 'success');
        } else {
            // Create
            await api.post('/gateway/containers', data);
            showToast('Kontainer baru berhasil ditambahkan!', 'success');
        }

        closeModal();
        loadContainers();
    } catch (error) {
        if (error.response?.status === 403) {
            showToast('Forbidden: Anda tidak memiliki hak akses.', 'error');
        } else if (error.response?.status === 422) {
            const errors = error.response.data.errors;
            const firstError = Object.values(errors)[0][0];
            showToast(firstError, 'error');
        } else {
            showToast('Gagal menyimpan kontainer.', 'error');
        }
    }
}

async function archiveContainer(id) {
    if (!confirm('Apakah Anda yakin ingin meng-archive kontainer ini?')) return;

    try {
        await api.patch(`/gateway/containers/${id}/archive`);
        showToast('Kontainer berhasil di-archive.', 'success');
        loadContainers();
    } catch (error) {
        if (error.response?.status === 403) {
            showToast('Forbidden: Hanya admin yang bisa meng-archive.', 'error');
        } else {
            showToast('Gagal meng-archive kontainer.', 'error');
        }
    }
}

async function deleteContainer(id) {
    if (!confirm('Apakah Anda yakin ingin menghapus kontainer ini? Data tracking log juga akan terhapus.')) return;

    try {
        await api.delete(`/gateway/containers/${id}`);
        showToast('Kontainer berhasil dihapus.', 'success');
        loadContainers();
    } catch (error) {
        if (error.response?.status === 403) {
            showToast('Forbidden: Hanya admin yang bisa menghapus.', 'error');
        } else {
            showToast('Gagal menghapus kontainer.', 'error');
        }
    }
}

// ==================== TRACKING LOGS ====================
async function viewTrackingLogs(containerId, containerCode) {
    document.getElementById('trackingModalTitle').textContent = `Log Perjalanan - ${containerCode}`;
    document.getElementById('trackingLogList').innerHTML = '<div class="empty-state"><div class="loading-spinner"></div><p style="margin-top:12px">Memuat log...</p></div>';
    document.getElementById('trackingModal').classList.add('active');

    try {
        const response = await api.get(`/gateway/containers/${containerId}/tracking-logs`);
        const logs = response.data.data;

        if (!logs || logs.length === 0) {
            document.getElementById('trackingLogList').innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-route"></i>
                    <h3>Belum ada log perjalanan</h3>
                </div>
            `;
            return;
        }

        document.getElementById('trackingLogList').innerHTML = logs.map(log => `
            <div style="background: var(--bg-input); border-radius: var(--radius-xs); padding: 16px; margin-bottom: 12px; border-left: 3px solid var(--primary);">
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <span style="font-size: 12px; color: var(--text-muted);">
                        <i class="fas fa-clock"></i> ${new Date(log.logged_at).toLocaleString('id-ID')}
                    </span>
                    ${log.status_change ? `<span class="type-badge">${escapeHtml(log.status_change)}</span>` : ''}
                </div>
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
                    <span style="color: var(--text-secondary);"><i class="fas fa-map-marker-alt" style="color: var(--danger);"></i> ${escapeHtml(log.location_from)}</span>
                    <i class="fas fa-arrow-right" style="color: var(--primary-light);"></i>
                    <span style="color: var(--text-primary);"><i class="fas fa-map-marker-alt" style="color: var(--success);"></i> ${escapeHtml(log.location_to)}</span>
                </div>
                ${log.notes ? `<p style="font-size: 13px; color: var(--text-secondary); margin-top: 8px;"><i class="fas fa-sticky-note" style="color: var(--accent);"></i> ${escapeHtml(log.notes)}</p>` : ''}
            </div>
        `).join('');
    } catch (error) {
        document.getElementById('trackingLogList').innerHTML = `
            <div class="empty-state">
                <i class="fas fa-exclamation-triangle"></i>
                <h3>Gagal memuat log</h3>
            </div>
        `;
    }
}

function closeTrackingModal() {
    document.getElementById('trackingModal').classList.remove('active');
}

// ==================== UTILITY FUNCTIONS ====================
function showToast(message, type = 'success') {
    const container = document.getElementById('toastContainer');
    const icons = {
        success: 'fa-check-circle',
        error: 'fa-exclamation-circle',
        warning: 'fa-exclamation-triangle',
    };

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `<i class="fas ${icons[type]}"></i> ${message}`;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(50px)';
        toast.style.transition = 'all 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 3500);
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
