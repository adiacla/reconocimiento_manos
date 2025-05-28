import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def analizar_csv_landmarks(csv_path):
    """
    Analiza un archivo CSV con landmarks para verificar su estructura.
    """
    if not os.path.exists(csv_path):
        print(f"[ERROR] Archivo no encontrado: {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path)
        print(f"\n=== ANÁLISIS DE {csv_path} ===")
        print(f"Dimensiones: {df.shape}")
        print(f"Primeras 5 columnas: {list(df.columns[:5])}")
        print(f"Últimas 5 columnas: {list(df.columns[-5:])}")

        # Analizar primera columna (etiquetas)
        print(f"\nEtiquetas en primera columna:")
        etiquetas_unicas = sorted(df.iloc[:, 0].unique())
        print(f"Valores únicos: {etiquetas_unicas}")
        print(f"Conteo por etiqueta:")
        conteo = df.iloc[:, 0].value_counts().sort_index()
        for etiqueta, cantidad in conteo.items():
            print(f"  Etiqueta {etiqueta}: {cantidad} muestras")

        # Verificar rango de los landmarks
        landmarks_cols = df.iloc[:, 1:64]  # Columnas 1-63
        print(f"\nRango de valores en landmarks:")
        print(f"  Mínimo: {landmarks_cols.min().min():.4f}")
        print(f"  Máximo: {landmarks_cols.max().max():.4f}")
        print(f"  Media: {landmarks_cols.mean().mean():.4f}")

        # Verificar valores faltantes
        valores_faltantes = df.isnull().sum().sum()
        print(f"\nValores faltantes: {valores_faltantes}")

        return True

    except Exception as e:
        print(f"[ERROR] Error analizando {csv_path}: {str(e)}")
        return False


def visualizar_landmarks_muestra(csv_path, num_muestra=0):
    """
    Visualiza los landmarks de una muestra específica.
    """
    try:
        df = pd.read_csv(csv_path)

        if num_muestra >= len(df):
            print(f"[ERROR] Muestra {num_muestra} no existe, máximo: {len(df)-1}")
            return

        # Extraer landmarks de la muestra
        muestra = df.iloc[num_muestra]
        etiqueta = muestra.iloc[0]
        landmarks = muestra.iloc[1:64].values

        # Separar coordenadas x, y, z
        x_coords = landmarks[:21]
        y_coords = landmarks[21:42]
        z_coords = landmarks[42:63]

        # Crear visualización
        fig = plt.figure(figsize=(15, 5))

        # Plot XY
        plt.subplot(1, 3, 1)
        plt.scatter(x_coords, y_coords, c=range(21), cmap="viridis")
        plt.title(f"Vista XY - Etiqueta: {etiqueta}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)

        # Plot XZ
        plt.subplot(1, 3, 2)
        plt.scatter(x_coords, z_coords, c=range(21), cmap="viridis")
        plt.title(f"Vista XZ - Etiqueta: {etiqueta}")
        plt.xlabel("X")
        plt.ylabel("Z")
        plt.grid(True)

        # Plot YZ
        plt.subplot(1, 3, 3)
        plt.scatter(y_coords, z_coords, c=range(21), cmap="viridis")
        plt.title(f"Vista YZ - Etiqueta: {etiqueta}")
        plt.xlabel("Y")
        plt.ylabel("Z")
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        print(f"Muestra {num_muestra} - Etiqueta: {etiqueta}")
        print(f"Rango X: [{x_coords.min():.3f}, {x_coords.max():.3f}]")
        print(f"Rango Y: [{y_coords.min():.3f}, {y_coords.max():.3f}]")
        print(f"Rango Z: [{z_coords.min():.3f}, {z_coords.max():.3f}]")

    except Exception as e:
        print(f"[ERROR] Error visualizando muestra: {str(e)}")


# Ejemplo de uso
if __name__ == "__main__":
    # Lista de archivos CSV para analizar
    archivos_csv = [
        "landmarks_extras/train_landmarks.csv",
        "landmarks_extras/valid_landmarks.csv",
        "datos_externos/hand_gestures_train.csv",
        "datos_externos/hand_gestures_valid.csv",
    ]

    print("=== ANALIZANDO ARCHIVOS CSV CON LANDMARKS ===")

    for csv_file in archivos_csv:
        if os.path.exists(csv_file):
            analizar_csv_landmarks(csv_file)

            # Preguntar si visualizar una muestra
            respuesta = input(f"\n¿Visualizar muestra de {csv_file}? (s/n): ")
            if respuesta.lower() == "s":
                num_muestra = input("Número de muestra (0 por defecto): ")
                try:
                    num_muestra = int(num_muestra) if num_muestra else 0
                    visualizar_landmarks_muestra(csv_file, num_muestra)
                except ValueError:
                    print("Número inválido, usando muestra 0")
                    visualizar_landmarks_muestra(csv_file, 0)
        else:
            print(f"[INFO] Archivo no encontrado (omitiendo): {csv_file}")

    print("\n=== ANÁLISIS COMPLETADO ===")
