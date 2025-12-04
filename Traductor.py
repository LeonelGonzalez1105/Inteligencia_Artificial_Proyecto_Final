import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print("--- 🧠 CARGANDO Y MEJORANDO DATASET MNIST ---")

# 1. Cargar datos
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. Normalizar y dar forma para CNN (necesita 4 dimensiones: batch, alto, ancho, canal)
x_train = x_train.reshape((-1, 28, 28, 1)) / 255.0
x_test = x_test.reshape((-1, 28, 28, 1)) / 255.0

# --- 🔥 INGENIERÍA: DATA AUGMENTATION ---
# Creamos un generador que deformará las imágenes al azar mientras entrena
datagen = ImageDataGenerator(
    rotation_range=10,      # Rotar hasta 10 grados
    zoom_range=0.1,         # Zoom de +/- 10%
    width_shift_range=0.1,  # Mover horizontalmente 10%
    height_shift_range=0.1  # Mover verticalmente 10%
)
datagen.fit(x_train)

print("--- 🏗️ CONSTRUYENDO MODELO PROFESIONAL (CNN) ---")
model = models.Sequential([
    # Capa Convolucional: Busca características como bordes y curvas
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)), # Reduce la imagen quedándose con lo importante
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Aplanamos y pasamos a las capas densas (como las de antes)
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3), # Apagamos 30% para evitar memorización
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("--- 🏋️ INICIANDO ENTRENAMIENTO INTENSIVO CON AUMENTO ---")
# Entrenamos usando el generador de datos
# Aumentamos a 10 epochs porque ahora el entrenamiento es más difícil
history = model.fit(datagen.flow(x_train, y_train, batch_size=32),
                    epochs=15, 
                    validation_data=(x_test, y_test))

print("--- 📊 EVALUANDO PRECISIÓN ---")
loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"Precisión del modelo: {accuracy*100:.2f}%")

model.save('ia_numeros.keras')
print("\n✅ ¡LISTO! Cerebro nuevo y mejorado guardado.")