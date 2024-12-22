import os
import cv2
import numpy as np

def train_recognizer(data_path, model_output_path='modelLBPHFace.xml'):
    """
    Entrena un reconocedor de rostros usando LBPH a partir de las imágenes en data_path.
    
    :param data_path: Ruta base donde se encuentran las carpetas de usuarios con sus imágenes de rostros.
    :param model_output_path: Nombre o ruta de salida para guardar el modelo entrenado.
    """
    
    print("Iniciando el entrenamiento...")
    people_list = os.listdir(data_path)

    labels = []
    faces_data = []
    label = 0

    print('Leyendo las imágenes...')
    for name_dir in people_list:
        person_path = os.path.join(data_path, name_dir)
        
        # Verificar que sea un directorio
        if not os.path.isdir(person_path):
            continue
        
        for file_name in os.listdir(person_path):
            file_path = os.path.join(person_path, file_name)
            # Lectura en escala de grises
            faces_data.append(cv2.imread(file_path, 0))
            labels.append(label)

        label += 1

    # Creando y entrenando el reconocedor LBPH
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    print("Entrenando el reconocedor de rostros...")
    face_recognizer.train(faces_data, np.array(labels))

    # Almacenando el modelo obtenido
    face_recognizer.write(model_output_path)
    print(f"Modelo LBPH almacenado en: {model_output_path}")
