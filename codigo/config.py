"""Constantes del proyecto: única fuente de verdad (semillas, rangos de K, umbrales y rutas).

Ningún otro módulo debe repetir estos valores; se importan desde aquí.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Rutas y nombres de archivos
# --------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parents[1]
DIR_CRUDOS = RAIZ / "datos" / "crudos"
DIR_PROCESADOS = RAIZ / "datos" / "procesados"
DIR_MODELOS = RAIZ / "modelos"

RUTA_CSV = DIR_CRUDOS / "Womens Clothing E-Commerce Reviews.csv"
RUTA_ZIP = DIR_CRUDOS / "archive.zip"
RUTA_FILTRADAS = DIR_PROCESADOS / "resenas_filtradas.csv"
RUTA_LIMPIAS = DIR_PROCESADOS / "resenas_limpias.csv"

ARCHIVO_VECTORIZADOR = "vectorizador.joblib"
ARCHIVO_SVD = "svd_lsa.joblib"
ARCHIVO_KMEANS = "modelo_kmeans.joblib"
ARCHIVOS_MODELO = (ARCHIVO_VECTORIZADOR, ARCHIVO_SVD, ARCHIVO_KMEANS)

# Nombres persistidos junto al modelo (formato {"0": "nombre", ...}) y la huella
# de la partición a la que pertenecen.
ARCHIVO_NOMBRES = "nombres_grupos.json"
ARCHIVO_HUELLA_NOMBRES = "nombres_grupos_huella.json"

# Decisión de K del agente (tabla del barrido, descartados, umbral y K elegido).
ARCHIVO_DECISION_K = "seleccion_k.json"

# --------------------------------------------------------------------------
# Reproducibilidad y entrenamiento
# --------------------------------------------------------------------------
SEMILLA = 42
N_INIT = 10                   # inicializaciones de K-means (se queda con la mejor)
N_COMPONENTES_SVD = 100
MUESTRA_SILUETA = 4000        # la silueta se calcula sobre una muestra de este tamaño
N_REPETICIONES_ESTABILIDAD = 10
N_INIT_ESTABILIDAD = 1        # n_init=1 a propósito: con 10 la estabilidad saldría inflada

# --------------------------------------------------------------------------
# Número de grupos (K)
# --------------------------------------------------------------------------
K_REFERENCIA_INFORME = 6      # K del modelo descrito en el informe
K_MIN = 2
K_MAX = 12
RANGO_K = range(K_MIN, K_MAX + 1)

# Regla de recomendación de K (ver agrupamiento.recomendar_k).
K_MIN_RECOMENDADO = 3
PROPORCION_MINIMA_GRUPO = 0.03
TOLERANCIA_SILUETA = 0.15

# --------------------------------------------------------------------------
# Triaje
# --------------------------------------------------------------------------
# Si la diferencia entre la distancia al 1.er y al 2.º grupo es menor que esto,
# la sugerencia se marca como "poco segura" (reseña de frontera).
MARGEN_DUDOSO = 0.02
N_TERMINOS_TOP = 10           # términos que describen cada grupo
N_SIMILARES = 3               # reseñas parecidas que se muestran con cada sugerencia

# --------------------------------------------------------------------------
# Animación de Lloyd (codigo/lloyd.py)
# --------------------------------------------------------------------------
MAX_ITER_LLOYD = 300          # como el valor por defecto de scikit-learn
TOL_LLOYD = 1e-4              # desplazamiento de centroides, relativo a la varianza media (como scikit-learn)

# --------------------------------------------------------------------------
# Presentación
# --------------------------------------------------------------------------
# Los grupos se numeran desde 1 SOLO al mostrarlos; modelos, huella, JSON e IDs internos usan 0.
PRIMER_GRUPO_VISIBLE = 1

# --------------------------------------------------------------------------
# Pestaña "Cómo el agente crea las categorías" (app/vista_creacion.py)
# --------------------------------------------------------------------------
# Velocidades de reproducción: nombre -> (pausa por paso, pausa por detalle) en segundos.
# Los cálculos son siempre reales; la pausa solo espacia lo que se muestra. ``None`` = ritmo
# real: se espera lo que tardó de verdad el cálculo de cada paso (los detalles no esperan).
VELOCIDADES: dict[str, tuple[float | None, float | None]] = {
    "Velocidad real": (None, None),
    "Lenta": (0.5, 0.1),
    "Muy lenta": (1.2, 0.25),
    "Didáctica": (2.5, 0.5),
}
VELOCIDAD_INICIAL = "Lenta"
MINIATURAS_POR_FILA = 4       # mosaico de la fase 1 (11 miniaturas -> 3 filas)
FILAS_MOSAICO = -(-len(RANGO_K) // MINIATURAS_POR_FILA)
MINIATURA_ALTO_PX = 120
TAMANO_PUNTO_MINIATURA = 2
MAPA_ALTO_PX = 560
SUBMUESTRA_MAPA = 1500        # el mapa y las miniaturas dibujan una muestra fija de este tamaño (medido: 4.675 puntos no son fluidos)
MAPA_WEBGL = True             # el mapa grande se dibuja con WebGL (miniaturas: SVG, para no agotar los contextos)
TAMANO_PUNTO_MAPA = 5
ITERACIONES_ESTELA = 5        # cuántas posiciones anteriores de cada centroide deja marcadas
PASOS_CONTEO = 6              # pasos con que sube el tamaño de cada grupo en la fase 3
LARGO_FRAGMENTO_RESENA = 140  # caracteres de la reseña que se muestran al pasar el cursor
ANCHO_LINEA_HOVER = 48
