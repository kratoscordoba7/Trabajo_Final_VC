import os
from capture_faces import capture_faces
from train_recognizer import train_recognizer
from recognize_faces import recognize_faces

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    data_path = os.path.join(script_dir, "userData")
    model_file = os.path.join(script_dir, "deploy.prototxt.txt")
    weights_file = os.path.join(script_dir, "res10_300x300_ssd_iter_140000.caffemodel")
    model_lbph_path = os.path.join(script_dir, "modelLBPHFace.xml")

    print("Bienvenido al Reconocedor Facial de Heliot y Alejandro!")
    print("Seleccione el modo de operación:")
    print("1. Modo completo: Captura, entrenamiento y reconocimiento.")
    print("2. Solo reconocimiento: Usar un modelo previamente entrenado.")
    
    mode = input("Ingrese el número del modo deseado (1 o 2): ")

    if mode == "1":
        # Modo completo
        person_name = input("Por favor, ingrese su nombre: ")

        # 1. Captura de rostros
        capture_faces(
            person_name=person_name,
            data_path=data_path,
            model_file=model_file,
            weights_file=weights_file
        )

        # 2. Entrenamiento de reconocimiento facial
        train_recognizer(data_path=data_path, model_output_path=model_lbph_path)

        # 3. Reconocimiento facial en tiempo real
        recognize_faces(
            data_path=data_path,
            model_file=model_file,
            weights_file=weights_file,
            model_lbph_path=model_lbph_path,
            frames_to_confirm=10  # Número de frames que deben coincidir
        )

    elif mode == "2":
        # Solo reconocimiento facial en tiempo real
        if not os.path.exists(model_lbph_path):
            print(f"Error: No se encontró el modelo entrenado en {model_lbph_path}.")
            print("Por favor, ejecute el modo completo para capturar y entrenar los datos.")
        else:
            recognize_faces(
                data_path=data_path,
                model_file=model_file,
                weights_file=weights_file,
                model_lbph_path=model_lbph_path,
                frames_to_confirm=10  # Número de frames que deben coincidir
            )
    else:
        print("Modo inválido. Por favor, ejecute nuevamente y seleccione una opción válida.")
