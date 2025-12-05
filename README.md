# Inteligencia_Artificial_Proyecto_Final
# Traductor Visual: Decimal a Binario con IA

Este proyecto utiliza **Deep Learning (TensorFlow)** y **Visión Artificial (OpenCV)** para reconocer números escritos a mano en tiempo real a través de una cámara web y traducirlos instantáneamente a su representación binaria.

## 🚀 ¿Cómo funciona?
1. **Captura:** OpenCV obtiene el video de la cámara en tiempo real.
2. **Preprocesamiento:** La imagen se convierte a escala de grises, se invierten los colores y se aplica una dilatación morfológica para mejorar el trazo y facilitar la lectura.
3. **Segmentación:** Se detectan contornos individuales y se ordenan de izquierda a derecha para leer cifras de varios dígitos (ej. "25").
4. **Inferencia:** Una Red Neuronal Convolucional (CNN), entrenada con el dataset MNIST y Data Augmentation, predice cada dígito.
5. **Traducción:** El sistema concatena los dígitos, convierte el número decimal a código binario y lo superpone en pantalla.

## 🛠️ Tecnologías Utilizadas
* **Lenguaje:** Python 3.10
* **IA Core:** TensorFlow / Keras (CNN)
* **Visión:** OpenCV (cv2)
* **Matemáticas:** NumPy

