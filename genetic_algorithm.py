import logging
import cv2
import numpy as np
from deap import base, creator, tools, algorithms
import json
import mysql.connector

logging.basicConfig(level=logging.INFO)
logging.getLogger("mysql.connector").setLevel(logging.WARNING)

##############################################
# Algoritmo genetico por torneo
##############################################

# Función de fitness
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.uniform, 0.0, 1.0)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=3)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Normalizar y limitar los pesos
def normalize(individual):
    total = sum(individual)
    if total > 1.0:
        return [x / total for x in individual]
    return individual

# Mutar para garantizar normalización
def mutate_and_normalize(individual):
    tools.mutGaussian(individual, mu=0, sigma=0.1, indpb=0.2)
    individual[:] = normalize(individual) 
    del individual.fitness.values  
    return individual,

toolbox.register("mutate", mutate_and_normalize)

# Cruzar para garantizar normalización
def mate_and_normalize(ind1, ind2):
    tools.cxBlend(ind1, ind2, alpha=0.5)
    ind1[:] = normalize(ind1)
    ind2[:] = normalize(ind2)
    del ind1.fitness.values  
    del ind2.fitness.values
    return ind1, ind2

toolbox.register("mate", mate_and_normalize)

cap = cv2.VideoCapture(0)

net = cv2.dnn.readNetFromCaffe(
    "./weights/deploy.prototxt.txt", 
    "./weights/res10_300x300_ssd_iter_140000.caffemodel"
)

face_recognizer_lbp = cv2.face.LBPHFaceRecognizer_create()
face_recognizer_lbp.read("./weights/lbph_ycrcb.xml")

face_recognizer_eigenfaces = cv2.face.EigenFaceRecognizer_create()
face_recognizer_eigenfaces.read("./weights/eigenfaces_noise_fourier.xml")

face_recognizer_fisherfaces = cv2.face.FisherFaceRecognizer_create()
face_recognizer_fisherfaces.read("./weights/fisherfaces_grayscale.xml")


def simulate_model_with_weights(individual):
    total_scores = []
    for _ in range(5):  # Procesar 5 frames para mayor robustez
        ret, frame = cap.read() 
        if not ret:
            logging.warning("No se pudo leer el frame de la cámara.")
            continue

        frame = cv2.resize(frame, (640, 480))
        h, w = frame.shape[:2]

        # Detectar el rostro
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

        # Evaluamos cada detección
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # Umbral de confianza
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x, y, x1, y1 = box.astype("int")
                x, y = max(0, x), max(0, y)
                x1, y1 = min(w - 1, x1), min(h - 1, y1)
                rostro = frame[y:y1, x:x1]
                rostro_gray = cv2.cvtColor(rostro, cv2.COLOR_BGR2GRAY)
                rostro_resized = cv2.resize(rostro_gray, (150, 150), interpolation=cv2.INTER_CUBIC)

                lbph_result_HSV = face_recognizer_lbp.predict(rostro_resized)
                fisher_result_GRAY = face_recognizer_fisherfaces.predict(rostro_resized)
                eigen_result_FOURIER = face_recognizer_eigenfaces.predict(rostro_resized)

                # Aplicamos mayor peso a LBPH y menor peso a Fisherfaces y Eigenfaces, porque LBPH es más preciso
                confidences = [
                    lbph_result_HSV[1] / 100,
                    fisher_result_GRAY[1] / 50000,
                    eigen_result_FOURIER[1] / 50000
                ]

                # Combinamos resultados usando los pesos
                total_score = sum(w * c for w, c in zip(individual, confidences))
                total_scores.append(total_score)
    return sum(total_scores) / len(total_scores) if total_scores else 0.0

# Función de fitness
def evaluate(individual):
    if sum(individual) > 1.0:
        individual[:] = normalize(individual)
    precision = simulate_model_with_weights(individual)
    return precision,

toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)


population = toolbox.population(n=20)
ngen = 20          # Número de generaciones
cruzamiento = 0.3  # Probabilidad de cruzamiento
mutacion = 0.1     # Probabilidad de mutación

result = algorithms.eaSimple(population, toolbox, cruzamiento, mutacion, ngen, verbose=True)
best_individual = tools.selBest(population, k=1)[0]

cap.release()
cv2.destroyAllWindows()

##############################################
# Guardamos los embeddings en la base de datos
##############################################

db_connection = mysql.connector.connect(
    host="localhost",
    user="root",     
    password="root", 
    database="datadb"
)

cursor = db_connection.cursor()
with open("./schemas/database.sql", "r", encoding="utf-8") as file:
    sql_script = file.read()

try:
    for statement in sql_script.split(';'):
        if statement.strip():  
            cursor.execute(statement)   
    db_connection.commit()
except mysql.connector.Error as err:
    print(f"Error: {err}")
    db_connection.rollback()

nombre = "Embeddings"
lbph = best_individual[0]
eigenfaces = best_individual[1]
fisherfaces = best_individual[2]

def guardar_embedding(nombre, lbph, eigenfaces, fisherfaces):
    delete_query = "DELETE FROM mi_tabla WHERE nombre = %s"
    cursor.execute(delete_query, (nombre,))
    db_connection.commit()

    lbph_json = json.dumps({"embedding": lbph})
    eigenfaces_json = json.dumps({"embedding": eigenfaces})
    fisherfaces_json = json.dumps({"embedding": fisherfaces})
    
    query = """
        INSERT INTO mi_tabla (nombre, lbph, eigenfaces, fisherfaces) 
        VALUES (%s, %s, %s, %s)
    """
    cursor.execute(query, (nombre, lbph_json, eigenfaces_json, fisherfaces_json))
    db_connection.commit()
    print(f"Embeddings guardados en la base de datos.")

guardar_embedding(nombre, lbph, eigenfaces, fisherfaces)