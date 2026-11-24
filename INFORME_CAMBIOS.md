# Informe de cambios: K elegido por el agente y creación visible de las categorías

Rama: `feature/seleccion-dinamica-clusters` · Commit base: `97d6185`.

**Verificación.** `python -m pytest -vv -s` en una ejecución limpia: **145 passed, 0 failed, 0 omitidas,
14 advertencias, 47.5 s** (con modelos entrenados). `entrenar_agente.py` se ejecutó (≈ 22 s) y la app
se probó con `AppTest` y en Chrome (mediciones de la sección 6).

## 1. Resumen

- **K ya no es fijo:** el agente barre K = 2..12, aplica una regla con tolerancia del 15 % y guarda el
  modelo y la decisión (`modelos/seleccion_k.json`). `--k N` lo impone a mano; `--barrido` se eliminó
  (el barrido siempre se hace).
- **Interfaz:** la barra lateral arranca con el K del agente y permite explorar otro K sin guardarlo.
  Hay dos pestañas nuevas: *Selección de K* y *Cómo el agente crea las categorías*.
- **Nombres atados a la partición** mediante una huella SHA-256 de la asignación de cada reseña.
- **Grupos numerados desde 1** para el usuario; internamente siguen siendo 0-based.
- Sin dependencias nuevas. No se tocaron preprocesamiento, TF-IDF, datos ni cuadernos.

## 2. Flujo automático de K

1. **Barrido** (`agrupamiento.barrido_k` / `iterar_barrido_k`): K-means por K, inercia, silueta sobre una
   muestra de 4 000 y tamaños de grupo.
2. **Regla** (`agrupamiento.recomendar_k`): descarta K < 3 y K con grupo menor < 3 % del corpus; entre los
   restantes toma la mejor silueta `s*` y elige el K más pequeño con silueta ≥ `s* − 0.15·|s*|`.
3. **Guardado** (`decision_k.py`): el modelo con ese K y `seleccion_k.json` (tabla, descartados, umbral,
   margen, semilla, fecha, origen `agente` o `manual`).

Tabla real (semilla 42):

| K | Inercia | Silueta | Grupo menor | Estado |
| --- | --- | --- | --- | --- |
| 2 | 944.68 | 0.0268 | 31.4 % | descartado: K menor que 3 |
| 3 | 931.45 | 0.0157 | 24.2 % | candidato |
| 4 | 920.60 | 0.0156 | 8.9 % | candidato |
| 5 | 911.66 | 0.0175 | 8.7 % | candidato |
| **6** | 903.62 | **0.0216** | 6.4 % | **elegido** |
| 7 | 896.25 | 0.0176 | 2.9 % | descartado: grupo más chico con 2.9 % (< 3 %) |
| 8 | 889.16 | 0.0184 | 2.995 % | descartado: grupo más chico con 3.0 % (< 3 %) |
| 9 | 881.04 | 0.0196 | 2.8 % | descartado: grupo más chico con 2.8 % (< 3 %) |
| 10 | 873.74 | 0.0192 | 3.3 % | candidato |
| 11 | 869.42 | 0.0249 | 3.2 % | candidato; mejor silueta |
| 12 | 863.06 | 0.0030 | 2.4 % | descartado: grupo más chico con 2.4 % (< 3 %) |

**Decisión:** K = 6. `s*` = 0.02492 (K = 11); umbral = 0.02492 − 0.15 · 0.02492 = **0.02118**;
silueta de K = 6 = 0.02161, margen **+0.00043** (≈ 2 % sobre el umbral): el K más pequeño que lo alcanza.

## 3. Archivos

| Archivo | Cambio |
| --- | --- |
| `codigo/config.py` | Única fuente de constantes: semilla, `RANGO_K`, umbrales de la regla, `N_INIT`, velocidades y tamaños de la animación, `SUBMUESTRA_MAPA`. |
| `codigo/agrupamiento.py` | `validar_k`, `iterar_barrido_k`, `barrido_k`, `recomendar_k`. |
| `codigo/decision_k.py` | **Nuevo.** `DecisionK`, `decidir_k`, `con_k_manual`, guardar/cargar JSON, `explicar_decision`. |
| `codigo/lloyd.py` | **Nuevo.** K-means paso a paso (réplica de scikit-learn con las mismas 10 inicializaciones), PCA 2D. |
| `codigo/creacion.py` | **Nuevo.** Eventos reales de la animación, con su duración de cálculo y el ritmo de reproducción. |
| `codigo/etiquetado.py` | Huella de partición, nombres persistidos con huella, nombre automático y nombre del informe. `HUELLA_REFERENCIA` registrada para K = 6. |
| `codigo/presentacion.py` | **Nuevo.** Numeración visible desde 1 (única fuente). |
| `codigo/triaje.py` | `con_kmeans` (otro K sin reentrenar TF-IDF ni SVD), huella y origen de nombres. |
| `app/` | App dividida: `estado.py`, `vista_triaje.py`, `vista_k.py`, `vista_creacion.py`, `escena_creacion.py`, `graficos.py`. |
| `entrenar_agente.py`, `triaje_cli.py` | Entrenar usa el barrido y la regla (`--k` impone); la terminal usa el K de `seleccion_k.json`. |
| `preparar_entorno.py` | **Nuevo.** Descarga los recursos de NLTK una vez; la app y los scripts avisan si faltan. |
| `pruebas/` | 145 pruebas (K, regla, decisión, huella, Lloyd, gráficos, creación, estado, scripts, `AppTest`). |

## 4. Numeración de grupos

La interfaz, la terminal y el CSV descargado muestran **Grupo 1 … Grupo K**; modelos, JSON y huella usan
0..K-1. Equivalencia con el informe anterior (que numeraba desde 0): **grupo del informe + 1**.

| Informe | Interfaz | Quejas | Nombre |
| --- | --- | --- | --- |
| 0 | 1 | 1 873 | Insatisfacción general de ajuste, color y tela |
| 1 | 2 | 435 | Calidad y precio |
| 2 | 3 | 361 | Expectativa vs. realidad |
| 3 | 4 | 300 | Silueta poco favorecedora |
| 4 | 5 | 720 | Tallaje |
| 5 | 6 | 986 | Apariencia vs. modelo |

## 5. Pestaña "Cómo el agente crea las categorías"

- **Fase 1:** 11 miniaturas (una por K, entrenado en vivo) y curva de silueta que se llena punto a punto;
  K descartados en gris con su motivo; al final, K elegido resaltado, explicación y comprobación contra
  `seleccion_k.json` (tabla igual y misma regla → "coincide").
- **Fase 2:** mapa 2D grande: quejas grises → semillas de k-means++ una a una → iteraciones con
  recoloreo, estela de centroides y contadores (iteración, inercia, quejas que cambiaron). Tira con las
  10 inicializaciones (inercia final, ARI frente al modelo; ganó la 10, ARI 1.00). Cursor con fragmento
  de reseña al terminar.
- **Fase 3:** rótulo de cada grupo en su centroide, términos uno a uno, tamaño que sube contando, nombre
  automático y nombre del informe.
- **Velocidades:** Velocidad real, Lenta (0,5 s/paso), Muy lenta (1,2 s) y Didáctica (2,5 s). **Para la
  demostración se recomienda "Lenta".** La primera ejecución calcula en vivo; después se repite sin
  recalcular. Los cálculos nunca se alteran: solo se espacian los pasos, y la pantalla lo dice.
- **Submuestra:** mapa y miniaturas dibujan **1 500 de las 4 675 quejas** (fija, semilla 42); los
  cálculos usan todas.
- **Avisos fijos:** el dibujo 2D explica ≈ 4.8 % de la varianza; su silueta (0.174) es la separación del
  dibujo 2D, no la calidad del modelo (0.0216 en 100 dimensiones); el modelo final es la mejor de 10
  inicializaciones y se anima esa ganadora.

## 6. Mediciones (Chrome con GPU, submuestra de 1 500 puntos)

| Velocidad | Fase 1 | Fase 2 | Fase 3 | Total |
| --- | --- | --- | --- | --- |
| Real, en vivo | 7.5 s | 3.2 s | < 10 ms | 10.8 s |
| Real, repetición | 7.3 s | 3.2 s | < 10 ms | 11.7 s |
| Lenta | 7.9 s | 25.5 s | 16.5 s | 50.6 s |
| Muy lenta | 18.6 s | 60.4 s | 40.0 s | 120.6 s |
| Didáctica | 38.2 s | 125.1 s | 80.8 s | 247.2 s |

Con los 4 675 puntos cada cuadro bloqueaba el navegador 200–260 ms (a "Lenta", 42 % del tiempo de la fase 2
en huecos > 50 ms). Con la submuestra bajó a 13 % (6 % a "Muy lenta", 3 % a "Didáctica"). A velocidad real
las iteraciones se ven cada ~150–200 ms: el dibujo es más lento que el cálculo (≈ 5.8 s la fase 1 y
≈ 2.7 s la fase 2 en el servidor).

## 7. Cómo reproducir

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python preparar_entorno.py
python entrenar_agente.py            # ≈ 22 s; --k N para imponer K
python -m pytest -vv -s              # 145 passed
streamlit run app/app_streamlit.py
```

## 8. Limitaciones conocidas

- **Silueta baja y margen estrecho:** 0.0216 es muy baja y K = 6 supera el umbral por solo +0.00043; las
  diferencias entre K pueden ser ruido de la muestra de silueta. La regla es heurística. K = 11 tiene
  mejor silueta pero grupos pequeños.
- **Vista 2D ≈ 4.8 % de la varianza:** los grupos se ven más mezclados de lo que están.
- **La huella cubre solo las asignaciones** (no centroides ni vectorizador). Si cambian por diferencias
  numéricas o del corpus, los nombres no se aplican (comportamiento buscado).
- **`nombres_grupos.json` antiguo** (sin `nombres_grupos_huella.json`) se ignora: volver a ejecutar
  `python entrenar_agente.py`.
- **`pytest -s`:** joblib/loky escribe en stderr una línea sobre núcleos físicos (`wmic` no existe en
  Windows 11). Es inofensiva.
- **Carga de CSV no cubierta por `AppTest`** (no puede subir archivos): se prueba `clasificar_lote` y la
  pestaña solo hasta "sin archivo cargado".
- **Dependencia de scikit-learn:** las pruebas de guarda de `lloyd.py` reproducen la secuencia de
  inicializaciones de `KMeans`; con otra versión de scikit-learn que la fijada (1.5.2) pueden fallar a
  propósito.
- **Los modelos con otro K viven solo en memoria** y los IDs de grupo cambian con K.
- Pendientes: estabilidad (ARI entre semillas) por K, guardar modelos por K, actualizar los cuadernos
  para que usen `barrido_k`.
