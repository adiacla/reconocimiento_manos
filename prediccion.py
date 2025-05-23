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
    # Verificar que el vector tenga el tamaño correcto
    if len(vector) != 63:  # 21 puntos * 3 coordenadas
        raise ValueError(f"Vector debe tener 63 elementos, tiene {len(vector)}")

    entrada = scaler.transform([vector])
    pred = model.predict(entrada, verbose=0)
    clase_pred = np.argmax(pred[0])
    confianza = np.max(pred[0])

    # DEBUGGING: Mostrar información detallada
    print(f"=== DEBUG PREDICCIÓN ===")
    print(
        f"Predicción: {le.inverse_transform([clase_pred])[0]} (Confianza: {confianza:.3f})"
    )

    print(f"Top 3 predicciones:")

    # Mostrar las 3 predicciones más probables
    top_indices = np.argsort(pred[0])[-3:][::-1]
    for i, idx in enumerate(top_indices):
        clase_nombre = le.inverse_transform([idx])[0]
        probabilidad = pred[0][idx]
        print(f"  {i+1}. {clase_nombre}: {probabilidad:.3f}")

    print("========================\n")

    return le.inverse_transform([clase_pred])[0], confianza


# Webcam - CREAR SOLO UNA VEZ
cap = cv.VideoCapture(0)

# Verificar que la cámara se abrió correctamente
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara")
    sys.exit(1)

print("Presiona 'q', ESC o Ctrl+C para salir")
print(f"Clases disponibles en el modelo: {le.classes_}")

# Crear ventana una sola vez
window_name = "Reconocimiento de Señas"
cv.namedWindow(window_name, cv.WINDOW_AUTOSIZE)

# Contador para controlar frecuencia de debugging
frame_count = 0
debug_frequency = 30  # Mostrar debug cada 30 frames (aprox 1 seg)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo capturar frame.")
            break

        frame_count += 1

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

                # Hacer predicción (solo debuggear cada ciertos frames)
                try:
                    # Activar debugging solo cada ciertos frames para no saturar la consola
                    if frame_count % debug_frequency == 0:
                        print(f"\n>>> FRAME {frame_count} - DEBUGGING ACTIVADO <<<")
                        signo, confianza = predecir(vector)
                    else:
                        # Predicción sin debugging
                        entrada = scaler.transform([vector])
                        pred = model.predict(entrada, verbose=0)
                        clase_pred = np.argmax(pred[0])
                        confianza = np.max(pred[0])
                        signo = le.inverse_transform([clase_pred])[0]

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

                    # Mostrar también el frame count y si debug está activo
                    debug_texto = f"Frame: {frame_count} | Debug: {'ON' if frame_count % debug_frequency == 0 else 'OFF'}"
                    cv.putText(
                        frame_out,
                        debug_texto,
                        (10, 60),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )

                except Exception as e:
                    print(f"ERROR en predicción: {str(e)}")
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
            "Presiona Q para salir | D para debug continuo",
            (10, frame_out.shape[0] - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # MOSTRAR EN LA VENTANA EXISTENTE
        cv.imshow(window_name, frame_out)

        # Controles de teclado
        key = cv.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # 'q' o ESC
            break
        elif key == ord("d"):  # 'd' para activar debug continuo
            debug_frequency = 1 if debug_frequency == 30 else 30
            print(f"Debug frequency cambiado a: cada {debug_frequency} frames")

except KeyboardInterrupt:
    print("\nInterrumpido por usuario")
finally:
    # Limpiar recursos de forma más robusta
    cap.release()
    cv.destroyWindow(window_name)  # Destruir ventana específica
    cv.destroyAllWindows()
    cv.waitKey(1)  # Asegurar que se procesen los eventos
    print("Aplicación cerrada correctamente")
