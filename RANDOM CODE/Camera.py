import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Menekan pesan TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Menonaktifkan optimasi oneDNN jika menyebabkan masalah

import cv2 
import numpy as np
from deepface import DeepFace
import time
import threading # Untuk menjalankan DeepFace di thread terpisah

# --- Konfigurasi Awal ---
CAMERA_INDEX = 0 # Ubah jika Anda memiliki multiple kamera
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30 # Target FPS untuk kamera, aktual mungkin berbeda
RECOGNITION_ACTIONS = ['age', 'gender', 'emotion', 'race'] # Tambahkan 'race'
#nVaraible acak yand sedang didefiniskan oleh angka acak
# Variabel Global untuk Threading dan Hasil Analisis
face_analysis_results = [] # Akan menyimpan hasil analisis untuk setiap wajah
last_analysis_time = 0
analysis_interval = 1.0  # Analisis setiap X detik (dapat diubah dengan +/-)
analysis_in_progress = False # Flag untuk menandakan apakah analisis sedang berjalan
analysis_thread = None # Objek thread
#Inisiasi metode yang  beda
# --- Inisialisasi Kamera ---
print("Mencoba membuka kamera...")
# Mencoba backend yang umum digunakan terlebih dahulu
backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] # DSHOW sering lebih stabil di Windows
cap = None
for backend_api in backends:
    cap = cv2.VideoCapture(CAMERA_INDEX, backend_api)
    if cap.isOpened():
        print(f"Berhasil membuka kamera dengan backend: {backend_api}")
        break
if not cap or not cap.isOpened():
    print("Error: Tidak dapat mengakses kamera dengan backend apapun.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
# Beberapa kamera mungkin tidak mendukung MJPG, bisa coba hilangkan jika ada error
try:
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
except Exception as e:
    print(f"Tidak bisa set FOURCC ke MJPG: {e}. Melanjutkan dengan default.")

# --- Inisialisasi Detektor Wajah ---
# Gunakan detektor wajah yang lebih cepat (Haar Cascade)
face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
if not os.path.exists(face_cascade_path):
    print(f"Error: File Haar Cascade tidak ditemukan di {face_cascade_path}")
    print("Pastikan OpenCV terinstal dengan benar atau path haarcascade valid.")
    cap.release()
    exit()
face_cascade = cv2.CascadeClassifier(face_cascade_path)
MIN_FACE_SIZE = (70, 70) # Ukuran minimum wajah yang akan dideteksi, sesuaikan jika perlu

# Variabel untuk FPS Counter
fps_counter = 0
fps = 0
start_time_fps = time.time()

print("\nMemulai deteksi wajah...")
print(f"Analisis DeepFace akan dilakukan setiap {analysis_interval:.1f} detik.")
print(f"Fitur yang dianalisis: {', '.join(RECOGNITION_ACTIONS)}")
print("Kontrol:")
print("  ESC   : Keluar")
print("  + / = : Tingkatkan interval analisis (+0.2s)")
print("  -     : Turunkan interval analisis (-0.2s)")
print("Peringatan: Analisis pertama mungkin membutuhkan waktu lebih lama untuk memuat model.\n")

# --- Fungsi untuk Analisis Wajah di Thread Terpisah ---
def analyze_faces_threaded(frame_copy, faces_coords):
    global face_analysis_results, analysis_in_progress
    
    current_results = []
    if not faces_coords: # Tidak ada wajah untuk dianalisis
        face_analysis_results = []
        analysis_in_progress = False
        return

    for (x, y, w, h) in faces_coords:
        # Pastikan ROI tidak keluar batas frame
        y_start, y_end = max(0, y), min(frame_copy.shape[0], y + h)
        x_start, x_end = max(0, x), min(frame_copy.shape[1], x + w)
        face_roi = frame_copy[y_start:y_end, x_start:x_end]

        if face_roi.size == 0: # Jika ROI kosong
            continue

        try:
            # Lakukan analisis DeepFace pada ROI
            # Penting: enforce_detection=False karena kita sudah menyediakan ROI
            # Penting: detector_backend='skip' karena kita sudah mendeteksi wajah
            result = DeepFace.analyze(
                img_path=face_roi,
                actions=RECOGNITION_ACTIONS,
                enforce_detection=False,
                detector_backend='skip', # Sangat penting untuk performa
                silent=True
            )
            
            # DeepFace.analyze mengembalikan list, bahkan jika hanya satu wajah di ROI
            if result and isinstance(result, list):
                res = result[0]
                # Menyesuaikan gender agar lebih singkat
                gender = "Pria" if res.get('dominant_gender') == 'Man' else "Wanita"
                
                current_results.append({
                    'rect': (x, y, w, h),
                    'age': res.get('age', '?'),
                    'gender': gender,
                    'emotion': res.get('dominant_emotion', '?').capitalize(),
                    'race': res.get('dominant_race', '?').capitalize()
                })
        except Exception as e:
            # print(f"Error dalam analisis DeepFace untuk satu ROI: {str(e)}") # Bisa di-uncomment untuk debugging
            # Tetap tambahkan placeholder agar urutan tidak rusak jika ada error parsial
            current_results.append({
                'rect': (x, y, w, h), 'error': True
            })
            
    face_analysis_results = current_results # Update hasil global
    analysis_in_progress = False

# --- Loop Utama ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Frame tidak terbaca. Mencoba melanjutkan...")
        if not cap.isOpened(): # Coba buka lagi jika tertutup
            print("Kamera terputus. Mencoba menyambungkan kembali...")
            cap.open(CAMERA_INDEX)
        time.sleep(0.5) # Beri jeda sebelum mencoba lagi
        continue
        
    frame_copy_for_analysis = frame.copy() # Salin frame untuk analisis agar tidak terpengaruh drawing
    
    # Hitung FPS
    fps_counter += 1
    current_time_fps = time.time()
    if current_time_fps - start_time_fps >= 1.0:
        fps = fps_counter
        fps_counter = 0
        start_time_fps = current_time_fps
    
    # Konversi ke grayscale untuk deteksi wajah yang lebih cepat
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Deteksi wajah menggunakan Haar Cascade
    detected_faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1,    # Seberapa besar gambar di-rescale pada setiap skala gambar
        minNeighbors=5,     # Seberapa banyak tetangga setiap kandidat rectangle harus dipertahankan
        minSize=MIN_FACE_SIZE # Ukuran minimum objek yang mungkin
    )
    
    faces_coords_current_frame = [] # Koordinat wajah di frame saat ini
    for (x, y, w, h) in detected_faces:
        faces_coords_current_frame.append((x,y,w,h))
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) # Kotak hijau untuk deteksi Haar

    # Kontrol Analisis DeepFace berdasarkan Interval dan status thread
    current_time_analysis = time.time()
    if not analysis_in_progress and (current_time_analysis - last_analysis_time > analysis_interval):
        if len(detected_faces) > 0: # Hanya analisis jika ada wajah
            analysis_in_progress = True # Set flag sebelum memulai thread
            last_analysis_time = current_time_analysis
            
            # Buat list (x,y,w,h) untuk dikirim ke thread
            rois_to_analyze = list(detected_faces)

            # Pastikan thread sebelumnya selesai jika ada (seharusnya tidak terjadi dengan flag analysis_in_progress)
            if analysis_thread and analysis_thread.is_alive():
                analysis_thread.join() 

            print(f"Memulai analisis untuk {len(rois_to_analyze)} wajah...")
            analysis_thread = threading.Thread(target=analyze_faces_threaded, args=(frame_copy_for_analysis, rois_to_analyze))
            analysis_thread.start()

    # Tampilkan hasil analisis yang tersimpan (dari thread)
    if face_analysis_results:
        for face_data in face_analysis_results:
            if face_data.get('error'): # Jika ada error saat analisis wajah ini
                text_y_offset = face_data['rect'][1] - 10
                cv2.putText(frame, "Error Analisis", (face_data['rect'][0], text_y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                continue

            x, y, w, h = face_data['rect']
            
            # Teks informasi
            age_text = f"Usia: {face_data['age']}"
            gender_text = f"Gender: {face_data['gender']}"
            emotion_text = f"Emosi: {face_data['emotion']}"
            race_text = f"Ras: {face_data['race']}"
            
            # Posisi teks
            text_y = y - 10
            cv2.putText(frame, f"{gender_text}, {age_text}", (x, text_y - 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, f"{emotion_text}", (x, text_y - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, f"{race_text}", (x, text_y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

    # Tampilkan Informasi Umum di Layar
    # FPS
    cv2.putText(frame, f"FPS: {fps}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Status Deteksi & Analisis
    status_text = f"Wajah Terdeteksi: {len(detected_faces)} | Interval: {analysis_interval:.1f}s"
    cv2.putText(frame, status_text, (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 0), 2)
    
    analysis_status_text = "Status: Menganalisis..." if analysis_in_progress else "Status: Idle"
    cv2.putText(frame, analysis_status_text, (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 0) if not analysis_in_progress else (0,165,255), 2)

    # Petunjuk
    cv2.putText(frame, "ESC: Keluar | +/-: Interval Analisis", 
                (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Tampilkan frame
    cv2.imshow('Real-time Face Analysis | @Doci', frame)
    
    # Tangkap input keyboard
    key = cv2.waitKey(1) & 0xFF # Ambil 8 bit terakhir untuk kompatibilitas
    if key == 27:  # Tombol ESC
        print("Menutup aplikasi...")
        break
    elif key == ord('+') or key == ord('='):
        analysis_interval = min(5.0, analysis_interval + 0.2) # Batas atas interval 5 detik
        print(f"Interval analisis ditingkatkan menjadi: {analysis_interval:.1f}s")
    elif key == ord('-'):
        analysis_interval = max(0.3, analysis_interval - 0.2) # Batas bawah interval 0.3 detik
        print(f"Interval analisis dikurangi menjadi: {analysis_interval:.1f}s")

# --- Bersihkan ---
print("Membersihkan sumber daya...")
if analysis_thread and analysis_thread.is_alive():
    print("Menunggu thread analisis selesai...")
    analysis_thread.join() # Pastikan thread selesai sebelum keluar

cap.release()
cv2.destroyAllWindows()
print("Selesai.")