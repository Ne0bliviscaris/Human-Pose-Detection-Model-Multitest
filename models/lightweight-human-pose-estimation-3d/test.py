import os

import cv2
import numpy as np
import torch


# Funkcja do ładowania modelu
def load_model(model_path):
    model = torch.load(model_path, map_location=torch.device("cpu"))  # CPU dla uproszczenia
    model.eval()  # Ustaw tryb ewaluacji
    return model


# Funkcja do przetwarzania jednej klatki
def process_frame(frame, model):
    # Przeskaluj klatkę do rozmiaru akceptowanego przez model
    input_size = 256  # Ustawienia domyślne, dopasuj je do wymagań modelu
    input_blob = cv2.dnn.blobFromImage(frame, 1 / 255, (input_size, input_size), (0, 0, 0), swapRB=False, crop=False)

    # Konwertuj klatkę na tensor PyTorch
    input_tensor = torch.from_numpy(input_blob).float()
    input_tensor = input_tensor.permute(0, 3, 1, 2)  # Zamień osie (B, H, W, C) -> (B, C, H, W)

    # Przekaż klatkę przez model
    with torch.no_grad():
        output = model(input_tensor)

    # Zwróć wynik wnioskowania
    return output


# Funkcja do uruchomienia na pliku wideo
def run_pose_estimation_on_video(video_path, model_path):
    model = load_model(model_path)  # Załaduj model

    # Otwórz wideo za pomocą OpenCV
    cap = cv2.VideoCapture(video_path)

    # Przetwarzaj klatka po klatce
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Przetwarzaj klatkę za pomocą modelu
        output = process_frame(frame, model)

        # Tutaj możesz dodać funkcję do rysowania wyników na obrazie
        # Na przykład: narysuj kluczowe punkty ciała na klatce

        # Wyświetl klatkę
        cv2.imshow("Pose Estimation", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Zwolnij zasoby
    cap.release()
    cv2.destroyAllWindows()


# Ścieżki do pliku MP4 i modelu
os.chdir("models\\lightweight-human-pose-estimation-3d")
video_path = "..\\..\\inputs\\moj1_400.mp4"
model_path = "modules\\human-pose-estimation-3d.pth"

# Uruchomienie funkcji
run_pose_estimation_on_video(video_path, model_path)
