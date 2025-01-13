import os
import cv2
import numpy as np

def dar_instrucciones(engine):
    """El bot dará las instrucciones necesarias."""
    instrucciones = (
        f"Estamos realizando el entrenamiento facial, "
        "por favor espere."
    )
    engine.say(instrucciones)
    engine.runAndWait()

def preprocess_images(image_paths, scale_type):
    """
        Preprocesa las imágenes según el tipo de escala especificado.

        :param image_paths: Lista de rutas de imágenes.
        :param scale_type: Tipo de escala ('hsv', 'ycrcb', 'grayscale', 'noise_fourier').
        :return: Lista de imágenes preprocesadas.
    """
    preprocessed_images = []

    for path in image_paths:
        image = cv2.imread(path)

        if scale_type == 'grayscale':
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif scale_type == 'noise_fourier':
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            f = np.fft.fft2(gray_image)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
            gray_image = magnitude_spectrum.astype(np.uint8)
        else:
            raise ValueError("Tipo de escala desconocido: " + scale_type)

        preprocessed_images.append(gray_image)

    return preprocessed_images

def train_and_save_model(images, labels, model_type, output_path):
    """
        Entrena un modelo de reconocimiento facial y lo guarda.

        :param images: Lista de imágenes preprocesadas.
        :param labels: Lista de etiquetas correspondientes a las imágenes.
        :param model_type: Tipo de modelo ('lbph' o 'eigenfaces').
        :param output_path: Ruta donde se almacenará el modelo entrenado.
    """
    if model_type == 'lbph':
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    elif model_type == 'eigenfaces':
        recognizer = cv2.face.EigenFaceRecognizer_create()
    elif model_type == 'fisherfaces':
        recognizer = cv2.face.FisherFaceRecognizer_create()
    else:
        raise ValueError("Tipo de modelo desconocido: " + model_type)

    print(f"Entrenando modelo {model_type}...")
    recognizer.train(images, np.array(labels))
    recognizer.write(output_path)
    print(f"Modelo {model_type} guardado en: {output_path}")



def train_recognizer(data_path, engine=None):
    """
        Realiza el entrenamiento de un modelo de reconocimiento facial.
        :param data_path: Ruta base donde se almacenan todas las carpetas de usuarios.
        :param engine: Motor de texto a voz para dar instrucciones.
    """

    print("Iniciando el entrenamiento...")
    people_list = os.listdir(data_path)

     # Dar las instrucciones al usuario
    dar_instrucciones(engine)

    labels = []
    faces_data = []
    label = 0

    print('Leyendo las imágenes...')
    for name_dir in people_list:
        person_path = os.path.join(data_path, name_dir)        
        if not os.path.isdir(person_path):
            continue
        
        for file_name in os.listdir(person_path):
            file_path = os.path.join(person_path, file_name)
            faces_data.append(file_path)
            labels.append(label)

        label += 1

    preprocessed_ycrcb = preprocess_images(faces_data, 'grayscale')
    preprocessed_grayscale = preprocess_images(faces_data, 'grayscale')
    preprocessed_noise_fourier = preprocess_images(faces_data, 'noise_fourier')

    train_and_save_model(preprocessed_ycrcb, labels, 'lbph', './weights/lbph_ycrcb.xml')
    train_and_save_model(preprocessed_grayscale, labels, 'fisherfaces', './weights/fisherfaces_grayscale.xml')
    train_and_save_model(preprocessed_noise_fourier, labels, 'eigenfaces', './weights/eigenfaces_noise_fourier.xml')
    
    print("Entrenamiento finalizado")