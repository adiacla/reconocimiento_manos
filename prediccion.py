import joblib
import numpy as np
import tensorflow as tf
import cv2 as cv
import mediapipe as mp

# Cargar modelos
model = tf.keras.models.load_model('model_sign_language.keras')
le = joblib.load('label_encoder.jb')
scaler = joblib.load('scaler.jb')

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# Función para predecir
def predecir(vector):
    entrada = scaler.transform([vector])
    pred = model.predict(entrada)
    print(pred)
    return le.inverse_transform([np.argmax(pred)])[0]

# Webcam
cap = cv.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("No se pudo capturar frame.")
        continue

    img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    resultado = hands.process(img_rgb)
    frame_out = frame.copy()

    if resultado.multi_hand_landmarks:
        for hand_landmarks in resultado.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame_out, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

            vector = []
            for lm in hand_landmarks.landmark:
                vector.extend([lm.x, lm.y, lm.z])

            signo = predecir(vector)

            print(vector)
            print(signo)
            if resultado.multi_hand_landmarks:
                cv.putText(frame_out, f'Signo: {signo}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            else:
                cv.putText(frame_out, 'Mano no detectada', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


    cv.imshow('Reconocimiento de Señas', frame_out)

    if cv.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
