<h1 align="center">🤖Trabajo Final - VC Reconocimiento facial 24/25</h1>

<div align="center">
<img width="500px" src="https://github.com/user-attachments/assets/289b1a6a-7d16-42c2-8441-e4f4a2aa0509">
</div>

Se ha completado el **Trabajo final** para la asignatura **Visión por Computador**.  Reconocimiento Facial. Consiste en aplicar distintas tecnicas
de recocimiento facial para poder reconocer personas.

*Trabajo realizado por*:

[![GitHub](https://img.shields.io/badge/GitHub-Heliot%20J.%20Segura%20Gonzalez-darkblue?style=flat-square&logo=github)](https://github.com/kratoscordoba7)

[![GitHub](https://img.shields.io/badge/GitHub-Alejandro%20D.%20Arzola%20Saavedra%20-purple?style=flat-square&logo=github)](https://github.com/AlejandroDavidArzolaSaavedra)

## 🛠️ Tecnologías Utilizadas

[![Python](https://img.shields.io/badge/Python-%233776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![MySQL](https://img.shields.io/badge/MySQL-%234479A1?style=for-the-badge&logo=mysql&logoColor=white)](https://www.mysql.com/)

## 🛠️ Librerías Utilizadas

[![OpenCV](https://img.shields.io/badge/OpenCV-%230076A8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Imutils](https://img.shields.io/badge/Imutils-%A020F0?style=for-the-badge)](https://pypi.org/project/imutils/)
[![OS](https://img.shields.io/badge/OS-%232196F3?style=for-the-badge&logo=linux&logoColor=white)](https://en.wikipedia.org/wiki/Operating_system)
[![NumPy](https://img.shields.io/badge/NumPy-%23013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Deque](https://img.shields.io/badge/Deque-%23E34F26?style=for-the-badge&logo=python&logoColor=white)](https://docs.python.org/3/library/collections.html#collections.deque)
[![SciPy](https://img.shields.io/badge/SciPy-%23045A8D?style=for-the-badge&logo=scipy&logoColor=white)](https://scipy.org/)
[![Logging](https://img.shields.io/badge/Logging-%23FF0000?style=for-the-badge&logo=python&logoColor=white)](https://docs.python.org/3/library/logging.html)
[![pyttsx3](https://img.shields.io/badge/pyttsx3-%23013243?style=for-the-badge&logo=python&logoColor=white)](https://pyttsx3.readthedocs.io/en/latest/)

---

## 🚀 Cómo empezar

Para comenzar con el proyecto, sigue estos pasos:

> [!NOTE]  
> Debes de situarte en un environment configurado como se definió en el cuaderno de la práctica de [otsedom](https://github.com/otsedom/otsedom.github.io/blob/main/VC/P1/README.md#111-comandos-basicos-de-anaconda).

### Paso 1: Abrir VSCode y situarse en el directorio:
   
   `C:\Users\TuNombreDeUsuario\anaconda3\envs\Trabajo_Final_VC
   
### Paso 2: Clonar y trabajar en el proyecto localmente (VS Code)
1. **Clona el repositorio**: Ejecuta el siguiente comando en tu terminal para clonar el repositorio:
   ```bash
   git clone https://github.com/kratoscordoba7/Trabajo_Final_VC.git
   ```
2. Una vez clonado, todos los archivos han de estar situado en el environment del paso 1

### Paso 3: Abrir Anaconda prompt y activar el environment:
   ```bash
   conda activate NombreDeTuEnvironment
   ```

### Paso 4: Instalación
Para instalar estas librerías, ejecuta los siguientes comandos:

```bash
pip install opencv-contrib-python numpy scipy imutils pyttsx3
```

Tras estos pasos debería poder ejecutar el proyecto localmente

<h2>📋 Motivación/argumentación del trabajo</h2>

<img align="left" width="200px" src="https://github.com/user-attachments/assets/7573a72a-704f-4b29-a32a-7dd498ed0183">  
Nuestro trabajo de curso se centra en el <b>reconocimiento facial</b>, una temática que nos despierta <b>curiosidad</b> 🧐 debido al funcionamiento y comportamiento de las <b>aplicaciones típicas nativas en dispositivos móviles</b> 📱 de la última década. 
<br><br>
Consideramos que el <b>reconocimiento facial</b> es un área <b>fascinante</b> ✨ que permite aplicar <b>técnicas y metodologías avanzadas de visión por computador</b> , ofreciendo una <b>oportunidad única</b> para explorar y desarrollar <b>soluciones innovadoras</b>. 💡
<br><br>

<h2>Alcance y objetivo de la propuesta</h2>


%%%%>


### Uso/Controles 📖💻

Una vez tengamos todas las librerías instaladas, debemos dirigirnos al archivo `main.py` y ejecutarlo. Para ello, basta con hacer clic en el botón indicado en la imagen siguiente: 

<div align="center">
   <img width="800px" src="https://github.com/user-attachments/assets/9fbc2553-0708-4b54-b8f7-93a1f19b5e54">
</div>

Al iniciar el programa, este nos dará la bienvenida y nos ofrecerá dos opciones para seleccionar. Estas opciones están diseñadas pensando en la flexibilidad del usuario:  

1️⃣ **Modo 1:** Capturar, entrenar y reconocer el rostro.  
2️⃣ **Modo 2:** Reconocer el rostro directamente, sin necesidad de un entrenamiento previo.  

Esta separación permite que el usuario pueda elegir lo que necesita sin tener que seguir pasos secuenciales innecesarios. Si solo deseas reconocer tu rostro sin haberlo entrenado antes, ¡no hay problema! Puedes seleccionar directamente el Modo 2.  

---

#### **Modo 1: Captura y Entrenamiento 🎥✨**

Si el usuario introduce "1", el programa solicitará un nombre que se utilizará para etiquetar correctamente el rostro durante el reconocimiento. Por ejemplo:  

<img src="/assets/img/modo1.gif">

Tras escribir el nombre, se abrirá una ventana con la cámara. En esta ventana, se indicará que al pulsar la tecla **"S"**, comenzará la captura del rostro. Para obtener mejores resultados en el reconocimiento, se recomienda:  

- Mirar directamente a la cámara.  
- Girar la cabeza suavemente hacia los lados.  

Aquí puedes ver un ejemplo del proceso:  

<div align="center">
   <img src="/assets/img/modo1_parte2.gif">
</div>

#### **Modo 2: Reconocimiento Directo 🔍🤖**

Si el usuario introduce "2", el programa intentará reconocer el rostro directamente.  

- Si logra identificar el rostro, aparecerá etiquetado con el nombre previamente entrenado.  
- Si no logra identificarlo, el sistema mostrará la etiqueta **"Desconocido"**.  

Aquí tienes una demostración del funcionamiento:  

<div align="center">
   <img src="/assets/img/modo2.gif">
</div>

###  Descripción técnica del trabajo realizado

Se enfatizan los aspectos más relevantes del proyecto.

La clase FaceIdentity está diseñada para representar una identidad única de un rostro detectado en un sistema de seguimiento. Rastrear y gestionar la información de un rostro detectado, asociándolo con un identificador único y usando un tracker para realizar seguimiento entre frames.

```Python
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

```

El siguiente fragmento utiliza el algoritmo de asignación de Hungarian para resolver la asociación entre las detecciones de rostros actuales y las identidades previamente rastreadas. Es crucial porque:

```Python
# Resolver la asignación óptima
row_ind, col_ind = linear_sum_assignment(cost_matrix)

# Asignar detecciones a identidades
for row, col in zip(row_ind, col_ind):
    distance = cost_matrix[row][col]
    if distance < max_distance:
        identities[row].centroid = detected_centroids[col]
        identities[row].missing_frames = 0
        assigned_identities.add(row)
        assigned_detections.add(col)
```

1. **Manejo de múltiples rostros**:
   - Al usar una matriz de costos (distancias entre centroides de las detecciones actuales y las identidades rastreadas), permite asociar de manera óptima los rostros detectados con sus correspondientes identidades en el menor costo posible.

2. **Rendimiento y precisión**:
   - Evita errores como asignaciones incorrectas o duplicadas mediante el cálculo preciso de distancias y la comparación con un umbral (`max_distance`).

3. **Escalabilidad**:
   - El enfoque permite que el sistema maneje múltiples rostros en tiempo real, algo fundamental en aplicaciones de reconocimiento facial.

4. **Prevención de errores**:
   - Si una detección no cumple con el criterio de distancia, no se asigna, reduciendo la probabilidad de errores en la identificación.

---

### 📂 **Estructura del Proyecto**  

```
   📜 `main.py` Programa principal que coordina la ejecución del proyecto.  
   📸 `capture_faces.py` Script encargado de la captura de rostros utilizando SSD (Single Shot Detector).  
   🤖 `train_recognizer.py` Script que realiza el entrenamiento del modelo** para el reconocimiento facial.  
   🕵️ `recognize_faces.py` Script encargado del reconocimiento facial a partir de los datos entrenados.  
   📂 `userdata/`  Carpeta que almacena todos los rostros capturados** durante el proceso.  
```

Los demás archivos que encontrarás en el proyecto son autogenerados por el propio programa, o necesarios para el proceso de entrenamiento.

### Conclusiones y propuestas de ampliación, propuestas adicionales


%%%%>


### Indicación de herramientas/tecnologías con las que les hubiera gustado contar / Aspectos a mejorar

Nos hubiera encantado contar con herramientas que pudieran integrar OpenCV con el desarrollo web de manera fácil, para mejorar la UI e incluso implementarlas en una aplicación web. Otras herramientas, como dlib, que generan muchos conflictos, también nos hubiera gustado probarlas. Vimos que tienen mucho potencial, pero no pudimos probarlas adecuadamente debido a problemas de incompatibilidades.

### Reuniones del grupo

Hemos tenido varias reuniones durante las clases prácticas, tanto con el profesor como con el propio equipo de desarrollo. Además, cada semana, o casi siempre, nos notificamos sobre el estado y avance del desarrollo de la aplicación.

### Créditos materiales no originales del grupo

Algunas imágenes utilizadas en el README y en el documento fueron generadas por DALL-E 3, y el video promocional fue desarrollado por Vidnoz.

### Entrenamiento

En cuanto al entrenamiento, utilizamos diferentes métodos como LBPH, Eigenfaces y Fisherfaces. Además, contamos con un proceso de entrenamiento que trabaja con un vector de descriptores, al cual le aplicamos un sumatorio y una ponderación en función de su desempeño. Estos descriptores son optimizados mediante algoritmos genéticos para acercarlos lo más posible a la solución ideal.  Tiene un comportamiento similar al de una neurona, es decir, estamos construyendo una función a partir de los descriptores, que permite que estos se comporten de manera que faciliten el reconocimiento facial.

La fórmula matemática es la siguiente:

$$
\sum D_N \times AG_N + \ldots
$$


### Vídeo resumen de venta del trabajo

VIDEO

---

> [!IMPORTANT]  
> Los decisiones de las implementaciones realizadas han sido recomendadas por [otsedom](https://github.com/otsedom/otsedom.github.io/tree/main/VC).

---

## 📚 Fuentes y tecnologías utilizadas

1. [Opencv](https://docs.opencv.org/)
2. [Tutorial face]([https://docs.opencv.org/](https://docs.opencv.org/4.x/da/d60/tutorial_face_main.html#tutorial_face_eigenfaces))
3. [Algoritmo hungarian](https://en.wikipedia.org/wiki/Hungarian_algorithm)
4. [Algoritmos geneticos](https://anderfernandez.com/blog/algoritmo-genetico-en-python/)
5. [Reconocimiento facial](https://omes-va.com/face-recognition-python/)
6. [Hungarian](https://python.plainenglish.io/hungarian-algorithm-introduction-python-implementation-93e7c0890e15)


---

<img align="left" width="140" height="140" src="https://github.com/user-attachments/assets/ed3617f0-9c77-44b2-a15b-b48052b0c9a4"></a>

**Universidad de Las Palmas de Gran Canaria**  

EII - Grado de Ingeniería Informática  
Obra bajo licencia de Creative Commons Reconocimiento - No Comercial 4.0 Internacional
<br><br><br>

---
