# Informe de cambios: selección configurable de K

Rama: `feat/seleccion-dinamica-clusters` · Commit base: `97d6185` · Cambios sin commit.

> **Estado de verificación.** Los cambios, incluidas las correcciones de la sección 10,
> están implementados pero **no se ejecutaron**: no se corrió `pytest`, ni
> `entrenar_agente.py`, ni la app de Streamlit. La única verificación hecha es sintáctica
> (`ast.parse` de los seis archivos Python modificados o creados). En `modelos/` no hay
> modelos entrenados, así que nada de lo descrito aquí se ha visto funcionar todavía.
> Este informe no contiene resultados de ejecución ni métricas nuevas.

## 1. Resumen ejecutivo

- K se elige en Streamlit entre 2 y 12. Al cambiarlo solo se entrena K-means sobre
  el TF-IDF + SVD ya calculado del corpus. TF-IDF y SVD no se reentrenan, y cada
  modelo por K queda en memoria.
- Las quejas nuevas (individuales y por lote) se asignan con el K elegido. Clasificar
  una queja nunca reentrena nada.
- Hay una pestaña nueva, **Selección de K**:
  - barrido K = 2..12 con barra de progreso real;
  - gráficos de inercia y silueta con el K elegido marcado;
  - tabla del barrido;
  - una recomendación **heurística** con sus limitaciones a la vista;
  - comparación con el modelo guardado (ARI y tabla de contingencia).
- Los nombres de grupo se atan a una partición exacta mediante una **huella** (SHA-256 de
  la asignación de cada reseña). Se aplican los de `nombres_grupos.json` o los manuales del
  informe solo si la huella coincide. En cualquier otro caso se usan nombres formados con
  los términos (sección 10).
- No se agregaron dependencias. No se tocaron preprocesamiento, TF-IDF, datos, modelos
  ni cuadernos.

## 2. Archivos modificados

| Archivo | Propósito |
|---|---|
| `codigo/agrupamiento.py` | Constantes `K_MIN`, `K_MAX`, `RANGO_K`, `MUESTRA_SILUETA`. Nuevas funciones `validar_k`, `barrido_k` (tabla + modelos + callback de progreso) y `recomendar_k` (con `RecomendacionK`). `seleccionar_k` conserva su firma y ahora delega en `barrido_k`. `entrenar_kmeans` rechaza `k ≥ n_filas`. |
| `codigo/etiquetado.py` | `huella_particion`, `HUELLA_REFERENCIA` (por ahora `None`), `es_modelo_referencia` y `nombres_manuales_validos` por huella exacta. `guardar_nombres`/`cargar_nombres` (`nombres_grupos.json` + `nombres_grupos_huella.json`), `NombresPersistidos` y `elegir_nombres_manuales`, que decide el origen de los nombres. |
| `codigo/triaje.py` | `AgenteTriaje` guarda `X_svd_corpus`, `kmeans_referencia` y `etiquetas_referencia`. Método nuevo `con_kmeans(kmeans)`, que crea un agente con otro K reutilizando vectorizador, SVD y corpus. Lee `nombres_grupos.json` y lo aplica solo a la partición con la misma huella. Nuevos atributos `huella`, `origen_nombres`, `aviso_nombres` y `nombres_manuales`, y propiedad `k_referencia`. |
| `app/app_streamlit.py` | Selector de K en la barra lateral, caché de modelos y agentes por K, pestaña "Selección de K". En las pestañas existentes, el K aparece en el resultado y en el CSV del lote. La barra lateral y la pestaña de grupos indican el origen de los nombres y muestran el aviso si se ignoró `nombres_grupos.json`. |
| `entrenar_agente.py` | `--k` validado en 2..12. `--barrido` imprime además la recomendación heurística. Después de guardar los modelos carga `AgenteTriaje`, imprime la huella de la partición y guarda los nombres con su huella. Conserva los nombres de un `nombres_grupos.json` anterior si pertenecen a la misma partición. |
| `pruebas/test_seleccion_k.py` | **Nuevo.** Pruebas con datos sintéticos de validación, barrido, recomendación, huella y nombres persistidos (incluidas particiones artificiales con los mismos tamaños y distinta asignación). Pruebas del agente que se omiten si no hay modelos. |
| `README.md` | Sección "Número de grupos (K) en la interfaz", uso de `--barrido`/`--k`, y orden de prioridad de los nombres con su huella. |
| `INFORME_CAMBIOS.md` | Este informe. |

## 3. Funcionalidades implementadas

1. **Selección manual de K** (`st.slider`, 2..12, validada con `agrupamiento.validar_k`).
   El valor inicial es el K del modelo guardado.
2. **Inferencia con el K elegido** en las pestañas "Queja individual", "Lote CSV" y
   "Grupos descubiertos": distancias a los K centroides, distribución y términos por grupo.
3. **Barrido K = 2..12** bajo demanda (botón), con progreso real por cada K terminado.
4. **Gráficos** de inercia y silueta (matplotlib, ya declarado en `requirements.txt`)
   con líneas para el K seleccionado y el recomendado.
5. **Tabla del barrido**: K, inercia, silueta, grupo menor, grupo mayor, proporción del
   grupo menor, y marcas de "seleccionado" y "recomendado".
6. **Recomendación heurística** con un botón "Usar K = x". La selección manual se conserva.
7. **Comparación con el modelo guardado**: ARI entre particiones y tabla de contingencia
   normalizada por fila. Reutiliza `evaluacion.indice_rand_ajustado` y
   `evaluacion.comparar_etiquetas_manuales`.
8. **Nombres ligados a la partición**: `nombres_grupos.json` verificado por huella, luego los
   nombres del informe si la huella es la de referencia, y si no, nombres por términos.

## 4. Cómo funciona K configurable

1. `AgenteTriaje()` carga del disco el vectorizador, el SVD y el K-means guardado.
   Calcula una sola vez `X_corpus` (TF-IDF del corpus) y `X_svd_corpus` (proyección SVD).
2. Si el K elegido es el del modelo guardado, se usa ese modelo sin entrenar nada.
3. Si es otro K:
   - Se entrena `agrupamiento.entrenar_kmeans(X_svd_corpus, k)` (semilla 42, `n_init=10`,
     los mismos parámetros que el modelo final).
   - El modelo se guarda en un diccionario compartido (`st.cache_resource`, protegido
     con un `threading.Lock`).
   - `con_kmeans` crea un agente derivado: copia superficial que comparte vectorizador,
     SVD y corpus, y recalcula etiquetas del corpus, términos y nombres.
4. El barrido usa la misma función de entrenamiento, así que los modelos que produce
   son los mismos que se entrenarían por separado y se reutilizan al elegir cualquier K.
5. Nada se escribe en disco. Al reiniciar el proceso de Streamlit, los modelos con
   otros K se vuelven a entrenar cuando se pidan.

## 5. Recomendación automática (heurística)

`agrupamiento.recomendar_k(tabla, k_min=3, proporcion_minima=0.03, tolerancia=0.10)`:

1. Descarta K < 3, porque dos grupos no distinguen tipos de queja.
2. Descarta los K cuyo grupo más chico tiene menos del 3 % del corpus (grupos degenerados).
3. Entre los candidatos busca la mejor silueta `s*`.
4. Recomienda el **K más pequeño** con silueta ≥ `s* − 0.10·|s*|` (parsimonia).

**No se presenta como garantía de calidad.** Según la tabla ya guardada en el cuaderno 03
(no recalculada aquí), la silueta del corpus es baja (≈0.003–0.027) y casi plana. Por eso
maximizar la silueta sin más no es defendible: elegiría K=2. Los umbrales (3, 3 %, 10 %)
son decisiones de diseño, no valores optimizados. **No se sabe qué K recomendará la regla
con los datos reales hasta ejecutar el barrido.**

## 6. Integración con Streamlit

- La barra lateral tiene el selector de K. El resumen del modelo indica el origen de los
  nombres (archivo verificado, informe o términos) y, si se ignoró `nombres_grupos.json`,
  muestra el motivo.
- Pestañas: "Queja individual", "Lote CSV", "Grupos descubiertos" y **"Selección de K"** (nueva).
- Cachés:
  - `_cargar_agente` (agente base);
  - `_almacen` (modelos por K, agentes derivados y tabla del barrido).
  Los dos se comparten entre sesiones del mismo proceso.
- Progreso real:
  - `st.spinner` mientras se entrena un K individual;
  - `st.progress` actualizado por el callback de `barrido_k` al terminar cada K.
  No hay animaciones simuladas.
- El botón "Usar K = x" cambia el selector mediante un callback de `st.session_state`.

## 7. Riesgos, limitaciones y pendientes

- **Nada fue ejecutado** (ver el aviso inicial). Hay que verificar en particular:
  - que la `Styler` con `background_gradient` se renderiza en Streamlit 1.40.1;
  - el comportamiento del slider con `key` y el callback;
  - los tiempos reales del barrido y del entrenamiento de un K.
- **`HUELLA_REFERENCIA` no está registrada.** Hasta que el equipo la registre, los nombres
  del informe no se aplican automáticamente a ningún modelo (ver sección 10).
- **La huella es estricta:** si una sola reseña cambia de grupo, por diferencias numéricas
  entre plataformas o versiones de librerías, o cambia el CSV procesado, los nombres no se
  aplican y se ven los nombres por términos. Ese es el comportamiento buscado.
- **Mientras corre el barrido**, el `Lock` bloquea en otras sesiones el entrenamiento de
  otros K.
- **Los modelos con otros K solo viven en memoria**: se pierden al reiniciar la app.
- **Los IDs de grupo cambian con K**: el grupo "4" con K=8 no tiene relación con el grupo
  "4" con K=6.
- **Pendientes (no implementados):**
  - estabilidad (ARI entre semillas) por cada K dentro de la recomendación;
  - proyección 2D del corpus y de la queja nueva;
  - guardar en disco modelos por K;
  - HDBSCAN (fuera de alcance);
  - actualizar los cuadernos para que usen `barrido_k`;
  - visualizar un reentrenamiento por reseña. No se implementó porque el sistema no
    reentrena y ese requisito no está confirmado.

## 8. Pruebas que debe ejecutar otro integrante

1. `python -m pytest -q` **antes** de entrenar. Se ejecutan las pruebas sintéticas de
   `test_seleccion_k.py` y las 3 de preprocesamiento; las del agente se omiten.
2. `python entrenar_agente.py` y luego `python -m pytest -q` de nuevo:
   - las 4 pruebas originales del agente (K=6) deben seguir pasando;
   - también `TestAgenteConOtroK`.
3. `python entrenar_agente.py --barrido`. Revisar la tabla y la recomendación impresas,
   y comparar con la tabla del cuaderno 03.
4. Prueba manual de la app (`streamlit run app/app_streamlit.py`):
   - Con K=6: mismos grupos, nombres manuales y distancias que antes del cambio.
   - Cambiar a K=8: aparece el spinner y la pestaña de grupos muestra 8 grupos con nombres
     por términos. Una queja de ejemplo muestra 8 distancias.
   - Volver a K=8 después de pasar por otro K: no debe volver a entrenar (sin spinner).
   - "Selección de K" → calcular barrido: la barra avanza de 1/11 a 11/11, aparecen los
     gráficos, la tabla y la recomendación. El botón "Usar K = x" cambia el selector.
   - La comparación con K≠6 muestra el ARI y la tabla de contingencia, y cada fila suma 100 %.
   - El lote CSV incluye la columna K y el archivo descargado se llama `triaje_resultado_k{K}.csv`.
   - Con un texto sin vocabulario se sigue mostrando el aviso.
5. Opcional: `python entrenar_agente.py --k 13` debe terminar con un error de validación.
6. Nombres y huella (sección 10):
   - Tras entrenar deben existir `modelos/nombres_grupos.json` (mismo formato
     `{"0": "nombre", ...}`) y `modelos/nombres_grupos_huella.json`. El script imprime la huella.
   - Editar a mano un nombre en `nombres_grupos.json` y abrir la app: el nombre editado
     aparece y la barra lateral dice "guardados en modelos/nombres_grupos.json (huella verificada)".
   - Con ese mismo archivo, cambiar a otro K: los nombres editados **no** aparecen; se ven
     nombres por términos.
   - Volver a ejecutar `python entrenar_agente.py` sin cambios: si la huella impresa es la
     misma, el nombre editado se conserva ("Se conservan los nombres...").
   - Borrar solo `nombres_grupos_huella.json`: la app debe ignorar `nombres_grupos.json` y
     mostrar el aviso de formato anterior.
   - Alterar la huella en `nombres_grupos_huella.json`: la app debe ignorar los nombres y
     avisar que pertenecen a otra partición.
   - Escribir un JSON inválido en `nombres_grupos.json`: la app carga igual y muestra el aviso.
   - Entrenar el mismo modelo en dos equipos distintos y comparar las huellas impresas. Si
     difieren, `HUELLA_REFERENCIA` solo valdrá en el entorno donde se registró.

Haz copia de los archivos de `modelos/` antes de alterarlos y restáuralos después.

## 9. Instrucciones para reproducir y verificar

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest -q                      # 1. pruebas sin modelos
python entrenar_agente.py                # 2. genera modelos/ (K = 6)
python -m pytest -q                      # 3. pruebas completas
python entrenar_agente.py --barrido      # 4. opcional: tabla K = 2..12 + recomendación (vuelve a guardar el modelo K = 6)
streamlit run app/app_streamlit.py       # 5. verificación manual (sección 8)
```

Revisar los cambios con `git diff` y `git status` antes de hacer commit. No se creó ningún commit.

## 10. Correcciones posteriores a la revisión del diff

### 10.1 Identidad del modelo de referencia

**Problema:** `es_modelo_referencia` reconocía la partición K=6 solo por los tamaños de los
grupos. Dos particiones distintas pueden tener los mismos tamaños.

**Corrección:**
- `etiquetado.huella_particion(etiquetas)` calcula un SHA-256 de la asignación completa.
  Usa enteros de 64 bits *little-endian*, en el orden del corpus, con un prefijo que incluye
  el número de reseñas y K.
- Dos particiones con los mismos tamaños pero distinta asignación, o con los IDs
  intercambiados, tienen huellas distintas.
- `es_modelo_referencia` y `nombres_manuales_validos` comparan contra `HUELLA_REFERENCIA`.
- `TAMANOS_REFERENCIA` se eliminó.

**Decisión de diseño:** el repositorio no contiene ninguna copia de la partición de
referencia. Los cuadernos solo guardaron los `.joblib`, que no están versionados, y el CSV
procesado no tiene columna de grupo. Por eso la huella **no se pudo calcular** en esta sesión
(no se permitió entrenar). `HUELLA_REFERENCIA = None`, y en ese estado los nombres del
informe no se aplican a ningún modelo. Se eligió fallar de forma segura: sin evidencia
fiable no se asignan nombres.

**Para registrarla:**
1. Ejecutar `python entrenar_agente.py` (K = 6).
2. Comprobar con `pytest` que se reproducen los tamaños y las distancias de la Tabla 44.
3. Revisar que los términos de cada grupo concuerdan con `NOMBRES_CLUSTER`.
4. Copiar la huella impresa en `HUELLA_REFERENCIA` de `codigo/etiquetado.py`.

La huella se calcula siempre sobre la partición que ve `AgenteTriaje`: `predict` sobre
`svd.transform(vectorizador.transform(texto_limpio))` del CSV procesado. `entrenar_agente.py`
la obtiene cargando ese mismo agente, de modo que la huella impresa, la guardada y la
comprobada en la app se calculan con el mismo código.

### 10.2 Compatibilidad con `nombres_grupos.json`

**Problema:** `entrenar_agente.py` escribía `nombres_grupos.json`, pero `AgenteTriaje` había
dejado de leerlo.

**Corrección:**
- `nombres_grupos.json` **mantiene su formato** `{"0": "nombre", ...}`.
- Se agrega el archivo auxiliar `modelos/nombres_grupos_huella.json` con
  `{"huella_particion", "k", "n_resenas"}`. Lo escribe `etiquetado.guardar_nombres`.
- `AgenteTriaje` lee ambos archivos una sola vez (`cargar_nombres`). En cada partición,
  `elegir_nombres_manuales` decide en este orden:
  1. `nombres_grupos.json`, si su huella coincide con la partición y sus IDs están en
     `0..K-1` (origen `"archivo"`);
  2. `NOMBRES_CLUSTER`, si la huella es `HUELLA_REFERENCIA` (origen `"referencia"`);
  3. nombres por términos (origen `"terminos"`).
- Los agentes con otro K comprueban la misma huella, así que nunca heredan los nombres del
  modelo guardado (salvo que la partición sea idéntica).
- `entrenar_agente.py` conserva los nombres de un `nombres_grupos.json` anterior cuando la
  nueva partición es idéntica (misma huella). Así, reentrenar no borra nombres editados a mano.
- Los JSON mal formados no rompen la carga: se ignoran y se muestra un aviso.

**Cambio de compatibilidad imprescindible:** un `nombres_grupos.json` generado antes de
esta corrección, sin `nombres_grupos_huella.json`, **se ignora**, porque no hay forma de
saber a qué partición pertenece. La app lo indica con un aviso. Solución: volver a ejecutar
`python entrenar_agente.py`. Los dos archivos están en `modelos/`, que ya ignora git.

### 10.3 Pruebas agregadas en `pruebas/test_seleccion_k.py`

Escritas pero **no ejecutadas**.

- Particiones artificiales de 6 grupos con tamaños idénticos (3, 2, 2, 1, 1, 1): una con dos
  reseñas intercambiadas y otra con los IDs 1 y 2 renombrados. Ninguna se reconoce como la
  de referencia ni recibe nombres guardados para otra partición.
- Huella determinista e independiente del tipo de entero; cambia si cambia el corpus.
- Sin `HUELLA_REFERENCIA` nunca se aplican los nombres del informe.
- `guardar_nombres` conserva el formato de `nombres_grupos.json` y escribe la huella.
- Prioridad: archivo verificado, luego referencia, luego términos. Un archivo de otra
  partición no impide usar la referencia.
- Se ignoran el archivo del formato anterior (sin huella), los IDs fuera de rango y el JSON
  inválido (`ValueError` en `cargar_nombres`).
- Con modelos entrenados:
  - la huella del agente corresponde a su partición;
  - los nombres guardados corresponden al modelo cargado;
  - los nombres persistidos no pasan a un modelo con el mismo K entrenado con otra semilla;
  - el modelo de referencia usa los nombres del informe (se omite mientras
    `HUELLA_REFERENCIA` sea `None`).
