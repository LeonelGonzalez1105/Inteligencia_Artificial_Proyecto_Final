import cv2
import numpy as np
import tensorflow as tf

# 1. Cargar el modelo entrenado
print("Cargando cerebro digital...")
try:
    model = tf.keras.models.load_model('ia_numeros.keras')
    print("✅ Modelo cargado exitosamente.")
except:
    print("❌ ERROR: No se encuentra 'ia_numeros.keras'. Ejecuta primero el entrenamiento.")
    exit()

# 2. Iniciar la cámara
cap = cv2.VideoCapture(0)

# Configuración de la ventana
cap.set(3, 640)
cap.set(4, 480)

print("--- 🎥 CÁMARA INICIADA ---")
print("Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret: break

    # --- PREPROCESAMIENTO ---
    # Coordenadas del cuadro verde
    x1, y1, x2, y2 = 200, 100, 400, 300
    roi = frame[y1:y2, x1:x2]
    
    # a. Escala de Grises
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # b. Umbralizado e Inversión (Para que quede Blanco sobre Negro)
    # Ajusté un poco el 140 a 128 para que sea más agresivo con el contraste
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    
    # --- 🔥 TRUCO NUEVO: ENGROSAMIENTO (DILATACIÓN) ---
    # Esto hará que las líneas delgadas de la tablet se vean "gordas" como plumón
    kernel = np.ones((5, 5), np.uint8) 
    thresh_gordo = cv2.dilate(thresh, kernel, iterations=1)
    
    # c. Redimensionar a 28x28
    img_ia = cv2.resize(thresh_gordo, (28, 28))
    
    # d. Normalizar y dar forma
    img_ia = img_ia / 255.0
    img_ia = img_ia.reshape(1, 28, 28)
    
    # --- PREDICCIÓN ---
    prediccion = model.predict(img_ia, verbose=0)
    numero_detectado = np.argmax(prediccion)
    confianza = np.max(prediccion) * 100
    
    texto_binario = "BIN: " + bin(numero_detectado)[2:]
    
    # --- DIBUJAR EN PANTALLA ---
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Mostrar el "Ojo IA" (Ahora verás el número más gordo aquí)
    img_visual = cv2.resize(thresh_gordo, (100, 100))
    frame[0:100, 0:100] = cv2.cvtColor(img_visual, cv2.COLOR_GRAY2BGR)
    cv2.putText(frame, "Ojo IA", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Mostrar resultado si la confianza es decente
    if confianza > 50: # Bajé un poco la exigencia al 50%
        color = (0, 255, 0) # Verde si está seguro
    else:
        color = (0, 0, 255) # Rojo si duda
        
    # Siempre mostramos el número, pero cambiamos el color según la confianza
    cv2.putText(frame, f"Detectado: {numero_detectado}", (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # Mostrar el binario solo si está seguro
    if confianza > 50:
        cv2.putText(frame, texto_binario, (x1, y2 + 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
    else:
        cv2.putText(frame, "?", (x1, y2 + 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("Traductor Decimal -> Binario", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()