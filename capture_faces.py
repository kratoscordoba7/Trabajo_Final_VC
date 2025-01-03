import cv2
import os
import imutils

def dar_instrucciones(engine, person_name):
    """El bot dará las instrucciones necesarias."""
    instrucciones = (
        f"Bienvenido {person_name}, coloque su rostro frente a la pantalla durante 3 segundos, "
        "y gire la cabeza 90 grados hacia un lado y hacia el otro cada 3 segundos, por favor."
    )
    engine.say(instrucciones)
    engine.runAndWait()

def capture_faces(person_name, data_path, model_file, weights_file, engine):
    """
        Captura los rostros de la persona especificada y los guarda en la carpeta correspondiente.
        
        :param person_name: Nombre de la persona para asignar la carpeta donde se guardarán las imágenes.
        :param data_path: Ruta base donde se almacenan todas las carpetas de usuarios.
        :param model_file: Ruta al archivo .prototxt del modelo DNN.
        :param weights_file: Ruta al archivo .caffemodel del modelo DNN.
    """
    
    person_path = os.path.join(data_path, person_name)

    # Dar las instrucciones al usuario
    dar_instrucciones(engine, person_name)

    if not os.path.exists(person_path):
        print('Carpeta creada:', person_path)
        os.makedirs(person_path)

    if not os.path.exists(model_file) or not os.path.exists(weights_file):
        print("Error: No se encuentran los archivos del modelo en las rutas especificadas.")
        return

    net = cv2.dnn.readNetFromCaffe(model_file, weights_file)
    print("Por favor, colóquese en el centro de la cámara y presione 'S' para iniciar la captura de rostros.")
    print("Estableciendo cámara...")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    count = 0
    start = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=640)
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 177.0, 123.0),
            swapRB=False,
            crop=False
        )
        
        # Detección de rostros
        net.setInput(blob)
        detections = net.forward()

        if not start:
            cv2.putText(
                frame, 
                "Presione 'S' para comenzar.",
                (5, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
        else:
            cv2.putText(
                frame, 
                "Escaneando...", 
                (5, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (10, 193, 255), 
                3, 
                cv2.LINE_AA
            )

        # Si se ha presionado 'S', comenzar a guardar rostros
        if start:
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:  # Umbral de confianza
                    box = detections[0, 0, i, 3:7] * [w, h, w, h]
                    x, y, x1, y1 = box.astype("int")

                    cv2.rectangle(frame, (x, y), (x1, y1), (10, 193, 255), 3)

                    rostro = frame[y:y1, x:x1]
                    try:
                        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
                        cv2.imwrite(f"{person_path}/rostro_{count}.jpg", rostro)
                        count += 1
                    except Exception as e:
                        print(f"Error al guardar rostro: {e}")

        cv2.imshow('Frame', frame)
        k = cv2.waitKey(1)

        if k == 27:  # Esc para salir
            break
        if k == ord('S') or k == ord('s'):
            start = True
            print("¡Presione 'ESC' para abortar la captura en cualquier momento.")
            print("Espere mientras se capturan sus rostros...")

        if count >= 100:
            print("Se han capturado suficientes rostros.")
            break

    cap.release()
    cv2.destroyAllWindows()
