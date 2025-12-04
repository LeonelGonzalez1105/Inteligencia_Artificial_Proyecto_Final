import cv2
import numpy as np
import tensorflow as tf

# --- 1. CARGAR MODELO ---
print("Cargando cerebro digital (Nivel 2)...")
try:
    model = tf.keras.models.load_model('ia_numeros.keras')
    print("✅ Modelo cargado.")
except:
    print("❌ Error: No se encuentra 'ia_numeros.keras'.")
    exit()

# --- 2. FUNCIÓN DE INGENIERÍA: PREPROCESADO ---
def preparar_digito(imagen_recortada):
    """
    Toma un recorte de número (ej. un '1' alto y flaco) y lo pone 
    en un cuadro negro cuadrado de 28x28 sin deformarlo.
    """
    # Escala de grises y umbralizado (si no viene ya procesado)
    if len(imagen_recortada.shape) > 2:
        gray = cv2.cvtColor(imagen_recortada, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    else:
        thresh = imagen_recortada

    # Hacer el número más grueso (Dilatación) para que se parezca a MNIST
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    # Mantener la relación de aspecto (Aspect Ratio)
    alto, ancho = thresh.shape
    
    # Queremos meterlo en un cuadro de 20x20 (dejando 4px de margen)
    if alto > ancho: # Es más alto que ancho (ej. un 1)
        factor = 20.0 / alto
        nuevo_alto = 20
        nuevo_ancho = int(ancho * factor)
    else: # Es más ancho que alto
        factor = 20.0 / ancho
        nuevo_ancho = 20
        nuevo_alto = int(alto * factor)
    
    # Redimensionar seguro
    if nuevo_ancho <= 0: nuevo_ancho = 1
    if nuevo_alto <= 0: nuevo_alto = 1
    
    img_re = cv2.resize(thresh, (nuevo_ancho, nuevo_alto))
    
    # Crear lienzo negro de 28x28 (Padding)
    lienzo = np.zeros((28, 28), dtype=np.uint8)
    
    # Pegar el número en el centro del lienzo
    centro_x = 14 - (nuevo_ancho // 2)
    centro_y = 14 - (nuevo_alto // 2)
    
    # Asegurar límites
    offset_y = min(centro_y + nuevo_alto, 28)
    offset_x = min(centro_x + nuevo_ancho, 28)
    
    lienzo[centro_y:offset_y, centro_x:offset_x] = img_re[0:(offset_y-centro_y), 0:(offset_x-centro_x)]
    
    # Normalizar para la IA
    lienzo = lienzo / 255.0
    lienzo = lienzo.reshape(1, 28, 28)
    return lienzo

# --- 3. INICIAR CÁMARA ---
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

print("--- 🚀 SISTEMA NIVEL 2 INICIADO ---")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Definir zona de detección (ROI) más ancha para caber varios números
    x1, y1, x2, y2 = 100, 100, 540, 300 # Rectángulo ancho
    roi_color = frame[y1:y2, x1:x2].copy()
    
    # Preprocesamiento general del ROI
    gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
    # Umbralizado simple
    _, thresh_roi = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    
    # --- DETECCIÓN DE CONTORNOS (SEGMENTACIÓN) ---
    contornos, _ = cv2.findContours(thresh_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrar ruido (contornos muy pequeños) y guardar los válidos
    candidatos = []
    for c in contornos:
        area = cv2.contourArea(c)
        if area > 200: # Si es muy chico es basura/ruido
            x, y, w, h = cv2.boundingRect(c)
            candidatos.append((x, y, w, h))
    
    # --- ORDENAR DE IZQUIERDA A DERECHA ---
    # Python ordena tuplas por el primer elemento (x), que es lo que queremos
    candidatos.sort(key=lambda b: b[0])
    
    numero_completo_str = ""
    
    # --- ANALIZAR CADA DÍGITO ENCONTRADO ---
    for (x, y, w, h) in candidatos:
        # Extraer el pedacito de imagen que tiene solo un número
        # Agregamos un poco de margen extra para no cortar bordes
        roi_digito = thresh_roi[y:y+h, x:x+w]
        
        # Preparar para la IA (Padding y resize inteligente)
        img_ia = preparar_digito(roi_digito)
        
        # Predicción
        prediccion = model.predict(img_ia, verbose=0)
        clase = np.argmax(prediccion)
        confianza = np.max(prediccion)
        
        if confianza > 0.5:
            numero_completo_str += str(clase)
            # Dibujar caja alrededor de CADA número individual
            cv2.rectangle(roi_color, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(roi_color, str(clase), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # --- RESULTADOS FINALES ---
    texto_binario = "Esperando..."
    if len(numero_completo_str) > 0:
        try:
            numero_decimal = int(numero_completo_str)
            texto_binario = f"BIN: {bin(numero_decimal)[2:]}"
        except:
            pass

    # --- DIBUJAR EN PANTALLA ---
    # Pegar el ROI procesado de vuelta en el frame principal para verlo
    frame[y1:y2, x1:x2] = roi_color
    
    # Dibujar el marco verde de la zona de captura
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Mostrar textos
    cv2.putText(frame, f"Decimal: {numero_completo_str}", (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    cv2.putText(frame, texto_binario, (50, 400), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # Mostrar ventana pequeña de qué ve la cámara en blanco y negro (debug)
    debug_view = cv2.resize(thresh_roi, (200, 100))
    frame[0:100, 0:200] = cv2.cvtColor(debug_view, cv2.COLOR_GRAY2BGR)
    cv2.putText(frame, "Vision Maquina", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    cv2.imshow("Traductor Nivel 2 (Multi-Digito)", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()