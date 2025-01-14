import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# 1. Lade das trainierte Modell
model = load_model('emotion_detection_model.h5')

# 2. Emotionen definieren (basierend auf den 7 Klassen)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# 3. Kamera initialisieren
cap = cv2.VideoCapture(0)

while True:
    # 4. Lese ein Bild von der Kamera
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # 5. Konvertiere das Bild in Graustufen (weil das Modell Graustufenbilder erwartet)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 6. Erkenne Gesichter im Bild (Verwende OpenCVs Haarcascades für Gesichtsdetektion)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    # 7. Für jedes erkannte Gesicht:
    for (x, y, w, h) in faces:
        # Extrahiere das Gesicht aus dem Bild
        face_region = gray[y:y+h, x:x+w]
        
        # 8. Skaliere das Gesicht auf 48x48 Pixel
        face_resized = cv2.resize(face_region, (48, 48))
        
        # 9. Normalisiere und bereite es für das Modell vor
        face_normalized = face_resized / 255.0  # Normalisierung
        face_normalized = np.expand_dims(face_normalized, axis=-1)  # Channel dimension hinzufügen
        face_normalized = np.expand_dims(face_normalized, axis=0)  # Batch dimension hinzufügen
        
        # 10. Vorhersage der Emotion
        prediction = model.predict(face_normalized)
        emotion_index = np.argmax(prediction)
        emotion = emotion_labels[emotion_index]
        
        # 11. Zeichne das Ergebnis auf dem Bild
        color = (0, 255, 0)  # Grün für das Rechteck (optional)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    
    # 12. Zeige das Bild mit der Emotion
    cv2.imshow('Emotion Recognition', frame)
    
    # 13. Beende die Schleife, wenn der Benutzer 'q' drückt
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 14. Schließe die Kamera und Fenster
cap.release()
cv2.destroyAllWindows()
