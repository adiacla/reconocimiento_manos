import os
import cv2
import mediapipe as mp
import xml.etree.ElementTree as ET
import csv

def obtener_etiqueta(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    return root.find('object').find('name').text


def procesar_directorio(directorio, salida_csv):
    mp_hands_module = mp.solutions.hands
    manos = mp_hands_module.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.8)

    with open(salida_csv, 'w', newline='') as archivo_csv:
        escritor = csv.writer(archivo_csv)
        encabezado = [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)] + [f'z{i}' for i in range(21)] + ['label']
        escritor.writerow(encabezado)

        total = 0
        procesadas = 0
        fallidas = 0

        for archivo in os.listdir(directorio):
            if not archivo.lower().endswith(('.jpg', '.jpeg', '.png')):
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

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            resultado = manos.process(img_rgb)

            if resultado.multi_hand_landmarks:
                for landmarks in resultado.multi_hand_landmarks:
                    x_vals = [lm.x for lm in landmarks.landmark]
                    y_vals = [lm.y for lm in landmarks.landmark]
                    z_vals = [lm.z for lm in landmarks.landmark]
                    escritor.writerow(x_vals + y_vals + z_vals + [etiqueta])
                    procesadas += 1
            else:
                print(f"[INFO] Mano no detectada en: {archivo}")
                fallidas += 1

        print(f"[RESUMEN] Total: {total}, Procesadas: {procesadas}, Fallidas: {fallidas}")


procesar_directorio('signlanguagevoc/train', 'train_data.csv')
procesar_directorio('signlanguagevoc/valid', 'valid_data.csv')
procesar_directorio('signlanguagevoc/test', 'test_data.csv')
