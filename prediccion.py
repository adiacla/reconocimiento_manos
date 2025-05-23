import cv2 as cv
import mediapipe as mp
import numpy as np
import tensorflow as tf
import joblib
import signal
import sys


# Función para manejar cierre limpio
def signal_handler(sig, frame):
    print("\nCerrando aplicación...")
    cap.release()
    cv.destroyAllWindows()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

# Cargar modelos
model = tf.keras.models.load_model("model_sign_language.keras")
le = joblib.load("label_encoder.jb")
scaler = joblib.load("scaler.jb")

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


# Función para predecir
def predecir(vector):
    entrada = scaler.transform([vector])
    pred = model.predict(entrada, verbose=0)
    clase_pred = np.argmax(pred[0])
    confianza = np.max(pred[0])
    return le.inverse_transform([clase_pred])[0], confianza


# Webcam - CREAR SOLO UNA VEZ
cap = cv.VideoCapture(0)

# Verificar que la cámara se abrió correctamente
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara")
    sys.exit(1)

print("Presiona 'q', ESC o Ctrl+C para salir")

# Crear ventana una sola vez
window_name = "Reconocimiento de Señas"
cv.namedWindow(window_name, cv.WINDOW_AUTOSIZE)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo capturar frame.")
            break

        # Voltear horizontalmente para efecto espejo
        frame = cv.flip(frame, 1)

        img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        resultado = hands.process(img_rgb)
        frame_out = frame.copy()

        if resultado.multi_hand_landmarks:
            for hand_landmarks in resultado.multi_hand_landmarks:
                # Dibujar landmarks
                mp_drawing.draw_landmarks(
                    frame_out,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

                # Extraer vector de características
                x_vals = [lm.x for lm in hand_landmarks.landmark]
                y_vals = [lm.y for lm in hand_landmarks.landmark]
                z_vals = [lm.z for lm in hand_landmarks.landmark]
                vector = x_vals + y_vals + z_vals

                # Hacer predicción
                try:
                    signo, confianza = predecir(vector)

                    # Mostrar resultado solo si la confianza es alta
                    if confianza > 0.6:
                        texto = f"Signo: {signo} ({confianza:.2f})"
                        color = (0, 255, 0)  # Verde
                    else:
                        texto = f"Incierto: {signo} ({confianza:.2f})"
                        color = (0, 255, 255)  # Amarillo

                    cv.putText(
                        frame_out,
                        texto,
                        (10, 30),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2,
                    )

                except Exception as e:
                    cv.putText(
                        frame_out,
                        f"Error: {str(e)[:30]}",
                        (10, 30),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )
        else:
            cv.putText(
                frame_out,
                "Mano no detectada",
                (10, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        # Mostrar instrucciones
        cv.putText(
            frame_out,
            "Presiona Q para salir",
            (10, frame_out.shape[0] - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # MOSTRAR EN LA VENTANA EXISTENTE
        cv.imshow(window_name, frame_out)

        # Salir con 'q' o ESC
        key = cv.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # 'q' o ESC
            break

except KeyboardInterrupt:
    print("\nInterrumpido por usuario")
finally:
    # Limpiar recursos de forma más robusta
    cap.release()
    cv.destroyWindow(window_name)  # Destruir ventana específica
    cv.destroyAllWindows()
    cv.waitKey(1)  # Asegurar que se procesen los eventos
    print("Aplicación cerrada correctamente")
