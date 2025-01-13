import cv2
import os
import numpy as np
from collections import deque
from scipy.optimize import linear_sum_assignment
import logging
import json
import mysql.connector

# Configuración básica de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("recognize_faces.log"),
        logging.StreamHandler()
    ]
)

class FaceIdentity:
    def __init__(self, face_id, initial_centroid, frames_to_confirm, initial_box, frame):
        """
        Inicializa una nueva identidad de rostro.

        :param face_id: Identificador único de la identidad.
        :param initial_centroid: Tupla (x, y) del centroide inicial del rostro.
        :param frames_to_confirm: Número de frames para confirmar la etiqueta.
        :param initial_box: Tupla (x, y, x1, y1) de la caja inicial del rostro.
        :param frame: Frame actual para inicializar el tracker.
        """
        self.id = face_id
        self.centroid = initial_centroid
        self.labels = deque(maxlen=frames_to_confirm)
        self.confirmed_label = None
        self.missing_frames = 0  # Contador de frames donde no se ha detectado el rostro
        self.tracker = cv2.TrackerCSRT_create()
        x, y, x1, y1 = initial_box
        w, h = x1 - x, y1 - y
        self.tracker.init(frame, (x, y, w, h))
        logging.info(f"Creada nueva identidad: ID {self.id} en centroid {self.centroid}")

def euclidean_distance(pt1, pt2):
    """
    Calcula la distancia euclidiana entre dos puntos.

    :param pt1: Tupla (x, y) del primer punto.
    :param pt2: Tupla (x, y) del segundo punto.
    :return: Distancia euclidiana.
    """
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

def recognize_faces(
    data_path, 
    model_file, 
    weights_file, 
    model_lbph_path='lbph.xml',
    frames_to_confirm=30,  # Aumentado a 30 frames según tu solicitud
    max_distance=50,
    max_missing_frames=5,
    engine=None
):
    """
    Reconoce rostros en tiempo real utilizando un modelo LBPH.
    Antes de asignar una etiqueta, requiere que se repita la misma predicción 
    en 'frames_to_confirm' frames consecutivos.

    :param data_path: Ruta base donde se encuentran las carpetas de usuarios.
    :param model_file: Ruta al archivo .prototxt del modelo DNN.
    :param weights_file: Ruta al archivo .caffemodel del modelo DNN.
    :param model_lbph_path: Ruta donde se encuentra el modelo LBPH entrenado.
    :param frames_to_confirm: Número de frames consecutivos necesarios para confirmar la etiqueta.
    :param max_distance: Distancia máxima para considerar que un rostro corresponde a una identidad existente.
    :param max_missing_frames: Número máximo de frames permitidos sin detección antes de eliminar la identidad.
    """
    
    # Verificar rutas de los archivos
    if not os.path.exists(model_file) or not os.path.exists(weights_file):
        logging.error("No se encuentran los archivos del modelo DNN en las rutas especificadas.")
        return
    
    # Cargar nombres de usuarios (carpetas) para mapear IDs a nombres
    image_paths = [
        name for name in os.listdir(data_path) 
        if os.path.isdir(os.path.join(data_path, name))
    ]
    image_paths.sort()  # Mantener un orden consistente

    logging.info(f"Nombres de usuarios cargados: {image_paths}")

    # Cargar el reconocedor fisher
    face_recognizer_lbph = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer_lbph.read("./weights/lbph_ycrcb.xml")

    face_recognizer_fisher = cv2.face.FisherFaceRecognizer_create()
    face_recognizer_fisher.read("./weights/fisherfaces_grayscale.xml")
    
    face_recognizer_eigen = cv2.face.EigenFaceRecognizer_create()
    face_recognizer_eigen.read("./weights/eigenfaces_noise_fourier.xml")


    logging.info("LBPH, Eigenfaces, FisherFaces cargado correctamente.")


    # Cargar el modelo DNN para detección de rostros
    net = cv2.dnn.readNetFromCaffe(model_file, weights_file)
    logging.info("DNN cargado correctamente.")

    # Inicializar la cámara
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        logging.error("No se pudo abrir la cámara.")
        return

    logging.info("Cámara inicializada correctamente.")


    #############################
    # Si la persona ha entrenado con el archivo genetic_algorithm.py, los pesos cambiaran, en vez de ser estaticos
    #############################

    lbph_confidence = 0.7
    eigen_confidence = 0.2
    fisher_confidence = 0.1
    umbral_confianza_bd = 0.2

    # Establecer la conexión a la base de datos
    db_connection = mysql.connector.connect(
        host="localhost",
        user="root",     
        password="root", 
        database="datadb"
    )

    # Creamos un cursor para ejecutar sentencias SQL
    cursor = db_connection.cursor()

    # Lista para mantener las identidades de los rostros
    identities = []
    face_id_counter = 0  # Contador para asignar IDs únicos

    # Consultamos para obtener todos los registros
    query = "SELECT id, nombre, lbph, eigenfaces, fisherfaces FROM mi_tabla"
    cursor.execute(query)

    # Leemos los registros
    resultados = cursor.fetchall()

    for row in resultados:
        id, nombre, lbph_json, eigenfaces_json, fisherfaces_json = row
        lbph_data = json.loads(lbph_json)  # Convertimos el JSON a un diccionario
        eigenfaces_data = json.loads(eigenfaces_json)
        fisherfaces_data = json.loads(fisherfaces_json)

        lbph_embedding = np.array(lbph_data["embedding"])
        eigenfaces_embedding = np.array(eigenfaces_data["embedding"])
        fisherfaces_embedding = np.array(fisherfaces_data["embedding"])

        lbph_confidence =lbph_embedding
        eigen_confidence = eigenfaces_embedding
        fisher_confidence = fisherfaces_embedding
        umbral_confianza_bd = 0.4

    # Cerrar la conexión
    cursor.close()
    db_connection.close()


    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("No se pudo leer el frame de la cámara.")
            break

        # Redimensionar frame para mayor eficiencia
        frame = cv2.resize(frame, (640, 480))
        h, w = frame.shape[:2]

        # Preprocesar para el modelo DNN
        blob = cv2.dnn.blobFromImage(
            frame, 
            scalefactor=1.0, 
            size=(300, 300), 
            mean=(104.0, 177.0, 123.0), 
            swapRB=False, 
            crop=False
        )
        net.setInput(blob)
        detections = net.forward()

        # Lista para almacenar las cajas detectadas en este frame
        detected_faces = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.4:  # Umbral de confianza
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x, y, x1, y1 = box.astype("int")
                # Asegurarse de que las coordenadas estén dentro del frame
                x, y = max(0, x), max(0, y)
                x1, y1 = min(w - 1, x1), min(h - 1, y1)
                detected_faces.append((x, y, x1, y1))
                # Dibujar la caja detectada (opcional para visualización)
                cv2.rectangle(frame, (x, y), (x1, y1), (255, 0, 0), 2)

        # Calcular centroides de las detecciones actuales
        detected_centroids = []
        for (x, y, x1, y1) in detected_faces:
            cx = int((x + x1) / 2)
            cy = int((y + y1) / 2)
            detected_centroids.append((cx, cy))

        # Logging de detecciones y identidades
        logging.info(f"Rostros detectados en el frame: {len(detected_faces)}")
        logging.info(f"Centroides detectados: {detected_centroids}")
        logging.info(f"Total de identidades actuales: {len(identities)}")

        # Inicializar 'assigned_identities' al inicio de cada frame
        assigned_identities = set()

        # Asignación de detecciones a identidades existentes usando el Algoritmo de Hungarian
        if len(identities) > 0 and len(detected_centroids) > 0:
            # Crear una matriz de costos (distancias) entre identidades y detecciones
            cost_matrix = []
            for identity in identities:
                cost_row = []
                for centroid in detected_centroids:
                    cost = euclidean_distance(identity.centroid, centroid)
                    cost_row.append(cost)
                cost_matrix.append(cost_row)
            
            # Logging de la matriz de costos
            logging.info("Matriz de costos:")
            logging.info(cost_matrix)
            
            # Resolver la asignación óptima
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Crear conjuntos para rastrear las asignaciones
            assigned_detections = set()

            # Asignar detecciones a identidades
            for row, col in zip(row_ind, col_ind):
                distance = cost_matrix[row][col]
                logging.info(f"Identidad {identities[row].id} asignada a Detección {col} con distancia {distance}")
                if distance < max_distance:
                    identities[row].centroid = detected_centroids[col]
                    identities[row].missing_frames = 0
                    assigned_identities.add(row)
                    assigned_detections.add(col)
                else:
                    logging.warning(f"Distancia {distance} excede el umbral para la identidad {identities[row].id}")

            # Crear nuevas identidades para detecciones no asignadas
            for idx, centroid in enumerate(detected_centroids):
                if idx not in assigned_detections:
                    face_id_counter += 1
                    # Obtener la caja correspondiente a la detección
                    x, y, x1, y1 = detected_faces[idx]
                    new_identity = FaceIdentity(face_id_counter, centroid, frames_to_confirm, (x, y, x1, y1), frame)
                    identities.append(new_identity)
        elif len(detected_centroids) > 0:
            # No hay identidades existentes, crear una para cada detección
            for centroid in detected_centroids:
                face_id_counter += 1
                idx = detected_centroids.index(centroid)
                x, y, x1, y1 = detected_faces[idx]
                new_identity = FaceIdentity(face_id_counter, centroid, frames_to_confirm, (x, y, x1, y1), frame)
                identities.append(new_identity)

        # Incrementar missing_frames para identidades no actualizadas
        for idx, identity in enumerate(identities):
            if idx not in assigned_identities:
                identity.missing_frames += 1
                logging.info(f"Incrementado missing_frames para identidad {identity.id}: {identity.missing_frames}")

        # Eliminar identidades que han estado ausentes durante demasiados frames
        before_removal = len(identities)
        identities = [identity for identity in identities if identity.missing_frames <= max_missing_frames]
        after_removal = len(identities)
        if before_removal != after_removal:
            logging.info(f"Eliminadas {before_removal - after_removal} identidades por falta de detección.")

        # Procesar cada identidad para reconocimiento
        for identity in identities:
            try:
                # Encontrar la detección más cercana al centroid
                min_dist = float('inf')
                matched_face = None
                for (x, y, x1, y1) in detected_faces:
                    cx = int((x + x1) / 2)
                    cy = int((y + y1) / 2)
                    dist = euclidean_distance((cx, cy), identity.centroid)
                    if dist < min_dist:
                        min_dist = dist
                        matched_face = (x, y, x1, y1)

                if matched_face:
                    x, y, x1, y1 = matched_face

                    # Extraer y preprocesar rostro
                    rostro = frame[y:y1, x:x1]
                    rostro_gray = cv2.cvtColor(rostro, cv2.COLOR_BGR2GRAY)
                    rostro_resized = cv2.resize(rostro_gray, (150, 150), interpolation=cv2.INTER_CUBIC)
                                        
                    result_fisher = face_recognizer_fisher.predict(rostro_resized)
                    predicted_label_fisher, confidence_label_fisher = result_fisher[0], result_fisher[1]

                    result_eigen = face_recognizer_eigen.predict(rostro_resized)
                    predicted_label_eigen, confidence_label_eigen = result_eigen[0], result_eigen[1]

                    result_lbph = face_recognizer_lbph.predict(rostro_resized)
                    predicted_label_lbph, confidence_label_lbph = result_lbph[0], result_lbph[1]



                    logging.info(f"Resultado :{umbral_confianza_bd},  {((confidence_label_lbph < 70) * lbph_confidence) + ((confidence_label_eigen < 3800) * eigen_confidence) + ((confidence_label_fisher < 2500) * fisher_confidence)}")

                    if(((confidence_label_lbph < 70) * lbph_confidence) + ((confidence_label_eigen < 3800) * eigen_confidence) + ((confidence_label_fisher < 2500) * fisher_confidence) > umbral_confianza_bd):
                        label_name = image_paths[predicted_label_lbph]
                    else:
                        label_name = "Desconocido"

                    logging.info(f"lbph_result {confidence_label_lbph < 70}, eigen_result {confidence_label_eigen < 3800}, fisher_result {confidence_label_fisher < 2500}")
                    # Añadir al buffer de etiquetas solo si no está confirmada
                    if identity.confirmed_label is None:
                        identity.labels.append(label_name)

                        # Verificar si se ha confirmado la etiqueta
                        if len(identity.labels) == frames_to_confirm:
                            # Contar frecuencia de cada label
                            freq = {}
                            for lbl in identity.labels:
                                freq[lbl] = freq.get(lbl, 0) + 1
                            # Obtener la etiqueta más común
                            common_label = max(freq, key=freq.get)
                            common_count = freq[common_label]
                            
                            if common_count >= (frames_to_confirm * 0.7):  # Al menos el 70% de los frames
                                identity.confirmed_label = common_label
                                logging.info(f"Etiqueta confirmada para identidad {identity.id}: {common_label}")
                    
                    # Determinar el texto a mostrar
                    if identity.confirmed_label:
                        display_text = identity.confirmed_label
                        color = (0, 255, 0)  # Verde
                    elif len(identity.labels) == frames_to_confirm:
                        display_text = "Desconocido"
                        color = (0, 0, 255)  # Rojo
                    else:
                        display_text = "Analizando..."
                        color = (255, 255, 0)  # Amarillo

                    # Dibujar el rectángulo y el texto
                    cv2.rectangle(frame, (x, y), (x1, y1), color, 2)
                    cv2.putText(
                        frame, 
                        f"ID {identity.id}: {display_text}", 
                        (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, 
                        color, 
                        2, 
                        cv2.LINE_AA
                    )

            except Exception as e:
                logging.error(f"Error procesando rostro ID {identity.id}: {e}")

        cv2.imshow('Reconocimiento Facial', frame)
        k = cv2.waitKey(1)
        if k == 27:  # ESC para salir
            break

    cap.release()
    cv2.destroyAllWindows()
