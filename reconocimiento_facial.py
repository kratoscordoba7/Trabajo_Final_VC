import cv2
import os
import imutils
import numpy as np

###############
# 1 Parte: Captura de rostros
###############

# Solicitar el nombre de la persona
print(f"Bienvenido al Reconocedor Facial de Heliot y Alejandro!")
personName = input("Por favor, ingrese su nombre: ")
dataPath = 'userData'  # Ruta donde se almacenarán los datos
personPath = os.path.join(dataPath, personName)

if not os.path.exists(personPath):
    print('Carpeta creada:', personPath)
    os.makedirs(personPath)

# Obtén la ruta del directorio del script actual
scriptDir = os.path.dirname(os.path.abspath(__file__))

# Cargar el modelo de detección de rostros basado en DNN
modelFile = os.path.join(scriptDir, "deploy.prototxt.txt")
weightsFile = os.path.join(scriptDir, "res10_300x300_ssd_iter_140000.caffemodel")

if not os.path.exists(modelFile) or not os.path.exists(weightsFile):
    print("Error: No se encuentran los archivos del modelo en el directorio actual.")
    exit()

net = cv2.dnn.readNetFromCaffe(modelFile, weightsFile)

print("Por favor, colóquese en el centro de la cámara y presione 'S' para iniciar el reconocimiento.")
print("Estableciendo cámara...")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
count = 0
start = False  # Variable para controlar cuando se presiona 'S'

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=640)
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
    
    # Detectar rostros
    net.setInput(blob)
    detections = net.forward()

    # Mostrar mensaje de instrucción
    if not start:
        cv2.putText(frame, "Presione 'S' para comenzar.", 
                    (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Escaneando...", 
                    (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (10, 193, 255), 3, cv2.LINE_AA)

    if start:
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # Umbral de confianza
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (x, y, x1, y1) = box.astype("int")

                # Dibujar el rectángulo del rostro
                cv2.rectangle(frame, (x, y), (x1, y1), (10, 193, 255), 3)
                rostro = frame[y:y1, x:x1]
                try:
                    rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(f"{personPath}/rostro_{count}.jpg", rostro)
                    count += 1
                except Exception as e:
                    print(f"Error al guardar rostro: {e}")
    
    # Mostrar el frame
    cv2.imshow('Frame', frame)

    # Manejo de teclas
    k = cv2.waitKey(1)
    if k == 27:  # Esc para salir
        break
    if k == 83:  # Número de la tecla 'S'
        start = True
        print(f"¡Presione 'ESC' para abortar.")
        print(f"Espere mientras se captura su rostro...")
        
    if count >= 100:
        print("Se han capturado suficientes rostros.")
        break

cap.release()
cv2.destroyAllWindows()

###############
# 2 Parte: Entrenamiento de reconocimiento facial
###############

# Método para entrenar el reconocedor
def entrenar_reconocedor():
    print("Iniciando el entrenamiento...")
    peopleList = os.listdir(dataPath)

    labels = []
    facesData = []
    label = 0
    print('Leyendo las imágenes')
    for nameDir in peopleList:
        personPath = os.path.join(dataPath, nameDir)

        for fileName in os.listdir(personPath):
            labels.append(label)
            facesData.append(cv2.imread(os.path.join(personPath, fileName), 0))
        label = label + 1

    # Creando y entrenando el reconocedor LBPH
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Entrenando el reconocedor de rostros
    print("Entrenando...")
    face_recognizer.train(facesData, np.array(labels))

    # Almacenando el modelo obtenido
    face_recognizer.write('lbph.xml')
    print("Modelo almacenado...")

entrenar_reconocedor()

###############
# 3 Parte: Reconocimiento Facial en tiempo real
###############

dataPath = 'userData' 
imagePaths = os.listdir(dataPath)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Leyendo el modelo de reconocimiento facial
face_recognizer.read('lbph.xml')

# Obtener el directorio del script actual
scriptDir = os.path.dirname(os.path.abspath(__file__))

# Cargar el modelo DNN para detección de rostros
modelFile = os.path.join(scriptDir, "deploy.prototxt.txt")
weightsFile = os.path.join(scriptDir, "res10_300x300_ssd_iter_140000.caffemodel")

if not os.path.exists(modelFile) or not os.path.exists(weightsFile):
    print("Error: No se encuentran los archivos del modelo en el directorio actual.")
    exit()

net = cv2.dnn.readNetFromCaffe(modelFile, weightsFile)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocesar la imagen para el modelo DNN
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Umbral de confianza
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (x, y, x1, y1) = box.astype("int")

            rostro = frame[y:y1, x:x1]
            try:
                # Preprocesar rostro para reconocimiento
                rostro = cv2.cvtColor(rostro, cv2.COLOR_BGR2GRAY)
                rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
                result = face_recognizer.predict(rostro)

                # Mostrar resultado en la ventana
                if result[1] < 70:
                    cv2.putText(frame, '{}'.format(imagePaths[result[0]]), (x, y - 25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
                else:
                    cv2.putText(frame, 'Desconocido', (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 2)
            except Exception as e:
                print(f"Error procesando rostro: {e}")

    cv2.imshow('Frame', frame)
    k = cv2.waitKey(1)
    if k == 27:  # Presiona ESC para salir
        break

cap.release()
cv2.destroyAllWindows()