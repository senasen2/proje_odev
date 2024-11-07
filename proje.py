import cv2
import numpy as np
import gradio as gr

# Emoji resmini yükleme
emoji = cv2.imread('star-struck.png', -1)

# Haar Cascade dosyasını yükleme
face_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface.xml')

# Emoji'yi yüzün üzerine bindiren fonksiyon
def overlay_emoji(frame, emoji, x, y, w, h):
    emoji = cv2.resize(emoji, (w, h))  # Emoji boyutunu yüz boyutuna göre ayarla
    emoji_h, emoji_w = emoji.shape[:2]

    # Emoji'nin alfa kanalını alma (saydamlık)
    alpha_s = emoji[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    # Bölgeye göre resmi yerleştirme
    for c in range(0, 3):
        frame[y:y+emoji_h, x:x+emoji_w, c] = (alpha_s * emoji[:, :, c] +
                                              alpha_l * frame[y:y+emoji_h, x:x+emoji_w, c])
    return frame

# Web kamerasından görüntü yakalama
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Gri tonlamalı bir kopya oluşturuyoruz (Yüz tespiti için)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Yüzleri tespit et
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Her yüz için emoji yerleştir
    for (x, y, w, h) in faces:
        frame = overlay_emoji(frame, emoji, x, y, w, h)

    # Sonuçları göster
    cv2.imshow('Emoji ile Canlı Yüz Tespiti', frame)

    # 'q' tuşuna basarak çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kameradan çık
cap.release()
cv2.destroyAllWindows()
