import os
import cv2
import mediapipe as mp
import xml.etree.ElementTree as ET
import csv
import numpy as np
import random
from scipy import ndimage


def obtener_etiqueta(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    return root.find("object").find("name").text


def augmentar_imagen(img):
    """Aplica augmentations específicos para lenguaje de señas"""
    augmented_images = [img]  # Imagen original

    # 1. Rotación ligera (-15 a +15 grados)
    for angle in [-10, -5, 5, 10]:
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
        augmented_images.append(rotated)

    # 2. Ajustes de brillo y contraste
    for brightness in [-30, -15, 15, 30]:
        for contrast in [0.8, 1.2]:
            adjusted = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
            augmented_images.append(adjusted)

    # 3. Flip horizontal (importante para señas simétricas)
    flipped = cv2.flip(img, 1)
    augmented_images.append(flipped)

    # 4. Escalado ligero
    for scale in [0.9, 1.1]:
        h, w = img.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        scaled = cv2.resize(img, (new_w, new_h))

        if scale < 1:  # Padding si es más pequeña
            top = (h - new_h) // 2
            bottom = h - new_h - top
            left = (w - new_w) // 2
            right = w - new_w - left
            scaled = cv2.copyMakeBorder(
                scaled, top, bottom, left, right, cv2.BORDER_REFLECT
            )
        else:  # Crop si es más grande
            top = (new_h - h) // 2
            left = (new_w - w) // 2
            scaled = scaled[top : top + h, left : left + w]

        augmented_images.append(scaled)

    # 5. Añadir ruido gaussiano ligero
    noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
    noisy = cv2.add(img, noise)
    augmented_images.append(noisy)

    return augmented_images


def normalizar_landmarks(landmarks):
    """Normaliza landmarks para hacerlos invariantes a traslación y escala"""
    # Convertir a numpy array
    points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])

    # Centrar en la muñeca (landmark 0)
    wrist = points[0]
    points_centered = points - wrist

    # Normalizar por la distancia máxima
    max_dist = np.max(np.linalg.norm(points_centered, axis=1))
    if max_dist > 0:
        points_normalized = points_centered / max_dist
    else:
        points_normalized = points_centered

    return points_normalized


def procesar_directorio_augmented(directorio, salida_csv, aplicar_augmentation=True):
    mp_hands_module = mp.solutions.hands
    manos = mp_hands_module.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.7,  # Reducido para captar más variaciones
    )

    with open(salida_csv, "w", newline="") as archivo_csv:
        escritor = csv.writer(archivo_csv)
        encabezado = (
            [f"x{i}" for i in range(21)]
            + [f"y{i}" for i in range(21)]
            + [f"z{i}" for i in range(21)]
            + ["label"]
        )
        escritor.writerow(encabezado)

        total = 0
        procesadas = 0
        fallidas = 0

        for archivo in os.listdir(directorio):
            if not archivo.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            total += 1
            nombre_base = os.path.splitext(archivo)[0]
            imagen_path = os.path.join(directorio, archivo)
            xml_path = os.path.join(directorio, f"{nombre_base}.xml")

            if not os.path.exists(xml_path):
                print(f"[WARN] XML no encontrado para: {archivo}")
                fallidas += 1
                continue

            etiqueta = obtener_etiqueta(xml_path)
            img = cv2.imread(imagen_path)
            if img is None:
                print(f"[ERROR] Imagen no válida: {archivo}")
                fallidas += 1
                continue

            # Procesar imagen original y augmentadas
            if aplicar_augmentation:
                imagenes_para_procesar = augmentar_imagen(img)
            else:
                imagenes_para_procesar = [img]

            imagen_procesada = False
            for idx, img_proc in enumerate(imagenes_para_procesar):
                img_rgb = cv2.cvtColor(img_proc, cv2.COLOR_BGR2RGB)
                resultado = manos.process(img_rgb)

                if resultado.multi_hand_landmarks:
                    for landmarks in resultado.multi_hand_landmarks:
                        # Aplicar normalización
                        points_norm = normalizar_landmarks(landmarks)

                        x_vals = points_norm[:, 0].tolist()
                        y_vals = points_norm[:, 1].tolist()
                        z_vals = points_norm[:, 2].tolist()

                        escritor.writerow(x_vals + y_vals + z_vals + [etiqueta])
                        procesadas += 1
                        imagen_procesada = True

            if not imagen_procesada:
                print(f"[INFO] Mano no detectada en ninguna variación: {archivo}")
                fallidas += 1

        print(
            f"[RESUMEN] Total: {total}, Procesadas: {procesadas}, Fallidas: {fallidas}"
        )


# Procesar con augmentation para entrenamiento
print("Procesando dataset de entrenamiento con augmentation...")
procesar_directorio_augmented(
    "signlanguagevoc/train", "train_data_augmented.csv", aplicar_augmentation=True
)

# Procesar sin augmentation para validación y test (mantener datos originales)
print("Procesando dataset de validación...")
procesar_directorio_augmented(
    "signlanguagevoc/valid", "valid_data.csv", aplicar_augmentation=False
)

print("Procesando dataset de test...")
procesar_directorio_augmented(
    "signlanguagevoc/test", "test_data.csv", aplicar_augmentation=False
)
