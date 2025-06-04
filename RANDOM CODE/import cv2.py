import cv2
import numpy as np
from deepface import DeepFace

# Inisialisasi webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    try:
        # Analisis wajah menggunakan DeepFace
        results = DeepFace.analyze(
            img_path=frame,
            actions=['age', 'gender', 'emotion'],
            enforce_detection=True,
            detector_backend='mtcnn',  # Detektor terakurat
            silent=True
        )

        # Ambil hasil pertama (asumsi hanya satu wajah)
        result = results[0]
        age = result['age']
        gender = result['dominant_gender']
        emotion = result['dominant_emotion']
        region = result['region']

        # Gambar kotak dan teks hasil
        x, y, w, h = region['x'], region['y'], region['w'], region['h']
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        text = f"{gender} {age}yo, {emotion}"
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    except Exception as e:
        pass  # Lewati jika tidak ada wajah terdeteksi

    # Tampilkan hasil
    cv2.imshow('Face Analysis', frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan
cap.release()
cv2.destroyAllWindows()