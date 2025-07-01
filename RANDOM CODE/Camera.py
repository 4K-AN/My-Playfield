import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 

import cv2 
import numpy as np
from deepface import DeepFace
import time
import threading


CAMERA_INDEX = 0 
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30
RECOGNITION_ACTIONS = ['age', 'gender', 'emotion', 'race']

face_analysis_results = [] 
last_analysis_time = 0
analysis_interval = 1.0  
analysis_in_progress = False 
analysis_thread = None 

print("Mencoba membuka kamera...")

backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] 
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

try:
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
except Exception as e:
    print(f"Tidak bisa set FOURCC ke MJPG: {e}. Melanjutkan dengan default.")


face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
if not os.path.exists(face_cascade_path):
    print(f"Error: File Haar Cascade tidak ditemukan di {face_cascade_path}")
    print("Pastikan OpenCV terinstal dengan benar atau path haarcascade valid.")
    cap.release()
    exit()
face_cascade = cv2.CascadeClassifier(face_cascade_path)
MIN_FACE_SIZE = (70, 70) 


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


def analyze_faces_threaded(frame_copy, faces_coords):
    global face_analysis_results, analysis_in_progress
    
    current_results = []
    if not faces_coords: 
        face_analysis_results = []
        analysis_in_progress = False
        return

    for (x, y, w, h) in faces_coords:
      
        y_start, y_end = max(0, y), min(frame_copy.shape[0], y + h)
        x_start, x_end = max(0, x), min(frame_copy.shape[1], x + w)
        face_roi = frame_copy[y_start:y_end, x_start:x_end]

        if face_roi.size == 0: 
            continue

        try:
         
            result = DeepFace.analyze(
                img_path=face_roi,
                actions=RECOGNITION_ACTIONS,
                enforce_detection=False,
                detector_backend='skip', 
                silent=True
            )
            
        
            if result and isinstance(result, list):
                res = result[0]
            
                gender = "Pria" if res.get('dominant_gender') == 'Man' else "Wanita"
                
                current_results.append({
                    'rect': (x, y, w, h),
                    'age': res.get('age', '?'),
                    'gender': gender,
                    'emotion': res.get('dominant_emotion', '?').capitalize(),
                    'race': res.get('dominant_race', '?').capitalize()
                })
        except Exception as e:
       
            current_results.append({
                'rect': (x, y, w, h), 'error': True
            })
            
    face_analysis_results = current_results 
    analysis_in_progress = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Frame tidak terbaca. Mencoba melanjutkan...")
        if not cap.isOpened(): 
            print("Kamera terputus. Mencoba menyambungkan kembali...")
            cap.open(CAMERA_INDEX)
        time.sleep(0.5) 
        continue
        
    frame_copy_for_analysis = frame.copy() 
    
  
    fps_counter += 1
    current_time_fps = time.time()
    if current_time_fps - start_time_fps >= 1.0:
        fps = fps_counter
        fps_counter = 0
        start_time_fps = current_time_fps
    
  
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
 
    detected_faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1,  
        minNeighbors=5,    
        minSize=MIN_FACE_SIZE 
    )
    
    faces_coords_current_frame = [] 
    for (x, y, w, h) in detected_faces:
        faces_coords_current_frame.append((x,y,w,h))
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) 

  
    current_time_analysis = time.time()
    if not analysis_in_progress and (current_time_analysis - last_analysis_time > analysis_interval):
        if len(detected_faces) > 0: 
            analysis_in_progress = True 
            last_analysis_time = current_time_analysis
            
         
            rois_to_analyze = list(detected_faces)

           
            if analysis_thread and analysis_thread.is_alive():
                analysis_thread.join() 

            print(f"Memulai analisis untuk {len(rois_to_analyze)} wajah...")
            analysis_thread = threading.Thread(target=analyze_faces_threaded, args=(frame_copy_for_analysis, rois_to_analyze))
            analysis_thread.start()


    if face_analysis_results:
        for face_data in face_analysis_results:
            if face_data.get('error'): 
                text_y_offset = face_data['rect'][1] - 10
                cv2.putText(frame, "Error Analisis", (face_data['rect'][0], text_y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                continue

            x, y, w, h = face_data['rect']
            
            
            age_text = f"Usia: {face_data['age']}"
            gender_text = f"Gender: {face_data['gender']}"
            emotion_text = f"Emosi: {face_data['emotion']}"
            race_text = f"Ras: {face_data['race']}"
            
          
            text_y = y - 10
            cv2.putText(frame, f"{gender_text}, {age_text}", (x, text_y - 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, f"{emotion_text}", (x, text_y - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, f"{race_text}", (x, text_y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

   
    cv2.putText(frame, f"FPS: {fps}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
   
    status_text = f"Wajah Terdeteksi: {len(detected_faces)} | Interval: {analysis_interval:.1f}s"
    cv2.putText(frame, status_text, (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 0), 2)
    
    analysis_status_text = "Status: Menganalisis..." if analysis_in_progress else "Status: Idle"
    cv2.putText(frame, analysis_status_text, (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 0) if not analysis_in_progress else (0,165,255), 2)

   
    cv2.putText(frame, "ESC: Keluar | +/-: Interval Analisis", 
                (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
  
    cv2.imshow('Real-time Face Analysis | @Doci', frame)
    
   
    key = cv2.waitKey(1) & 0xFF 
    if key == 27: 
        print("Menutup aplikasi...")
        break
    elif key == ord('+') or key == ord('='):
        analysis_interval = min(5.0, analysis_interval + 0.2) 
        print(f"Interval analisis ditingkatkan menjadi: {analysis_interval:.1f}s")
    elif key == ord('-'):
        analysis_interval = max(0.3, analysis_interval - 0.2) 
        print(f"Interval analisis dikurangi menjadi: {analysis_interval:.1f}s")


print("Membersihkan sumber daya...")
if analysis_thread and analysis_thread.is_alive():
    print("Menunggu thread analisis selesai Kalau habis selesai maka harus ditutup dengan salam...")
    analysis_thread.join() 

cap.release()
cv2.destroyAllWindows()
print("Selesai.")