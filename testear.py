import os
import cv2
import mediapipe as mp

# Ruta a tus imágenes
directorio = 'signlanguagevoc/train'  # Cambia por 'valid' o 'test' si deseas

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

imagenes_fallidas = []

for archivo in os.listdir(directorio):
    if not archivo.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    imagen_path = os.path.join(directorio, archivo)
    img = cv2.imread(imagen_path)
    if img is None:
        print(f"No se pudo cargar la imagen: {archivo}")
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resultado = hands.process(img_rgb)

    if resultado.multi_hand_landmarks:
        for landmark in resultado.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img, landmark, mp_hands.HAND_CONNECTIONS)

        # Redimensionar la imagen una sola vez fuera del bucle
        img_mostrar = cv2.resize(img, (400, 400))
        cv2.imshow("Mano detectada", img_mostrar)

    else:
        print(f"❌ Mano NO detectada en: {archivo}")
        imagenes_fallidas.append(archivo)
        img_mostrar = cv2.resize(img, (400, 400))
        cv2.imshow("Sin mano", img_mostrar)

    key = cv2.waitKey(500)  # Espera 0.5 segundos entre imágenes
    if key == ord('q'):
        break

hands.close()
cv2.destroyAllWindows()

# Reporte
print(f"\nTotal imágenes procesadas: {len(os.listdir(directorio))}")
print(f"Imágenes sin mano detectada: {len(imagenes_fallidas)}")
print("Listado de fallidas:")
for img in imagenes_fallidas:
    print(f" - {img}")
