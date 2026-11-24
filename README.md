# Triaje de quejas de reseñas de e-commerce

Agente de triaje de quejas en reseñas de productos (Inteligencia Artificial,
FCyT-UMSS, Grupo 6). Área: aprendizaje no supervisado. Subárea: procesamiento
de lenguaje natural.

El agente recibe el texto de una queja y **sugiere el grupo de quejas al que
más se parece**, mostrando los términos del grupo y reseñas similares. No
resuelve la queja ni responde al cliente.

Dataset: **Women's E-Commerce Clothing Reviews** (Kaggle, CC0, 23 486 reseñas
en inglés). El texto no se traduce.

## Pipeline

```text
CSV (23 486) ──► criterio de queja: Rating<=2 OR Recommended=0 OR VADER<=-0.05 ──► 4 675 quejas
           ──► preprocesamiento (minúsculas, contracciones, limpieza, tokens,
               stopwords sin negaciones, lematización)
           ──► TF-IDF (1-2 gramas, 5 000 términos) ──► SVD truncada (100)
           ──► barrido K = 2..12 ──► regla de K (tolerancia 15 %) ──► K-means final
           ──► evaluación (silueta 0.0216, ARI 0.4014) ──► nombres ──► interfaz de triaje
```

## Estructura

```text
.
├── app/
│   ├── app_streamlit.py      # Interfaz (5 pestañas)
│   ├── estado.py             # Modelos por K, locks y K activo
│   ├── vista_triaje.py       # Queja individual, lote CSV, grupos descubiertos
│   ├── vista_k.py            # Selección de K
│   ├── vista_creacion.py     # "Cómo el agente crea las categorías" (pestaña)
│   ├── escena_creacion.py    # Marcos y dibujo de esa pestaña
│   └── graficos.py           # Figuras de Plotly (funciones puras)
├── codigo/
│   ├── config.py             # Constantes: semilla, rango de K, umbrales, velocidades
│   ├── preprocesamiento.py   # Carga, criterio de queja (VADER) y limpieza del texto
│   ├── caracteristicas.py    # TF-IDF
│   ├── agrupamiento.py       # SVD, barrido de K, regla de K y K-means
│   ├── decision_k.py         # Decisión del agente (seleccion_k.json) y su explicación
│   ├── lloyd.py              # K-means paso a paso y proyección PCA 2D
│   ├── creacion.py           # Eventos reales de la animación de la pestaña nueva
│   ├── evaluacion.py         # Silueta, ARI y contingencia
│   ├── etiquetado.py         # Términos top, nombres y huella de la partición
│   ├── presentacion.py       # Numeración de grupos visible (desde 1)
│   └── triaje.py             # AgenteTriaje: sugiere el grupo de una queja nueva
├── cuadernos/                # 01 exploratorio · 02 preprocesamiento · 03 agrupamiento · 04 evaluación
├── datos/                    # crudos/ (CSV de Kaggle) y procesados/
├── modelos/                  # se generan al entrenar (ver abajo)
├── pruebas/                  # pytest
├── preparar_entorno.py       # Descarga una vez los recursos de NLTK
├── entrenar_agente.py        # Entrena todo sin Jupyter (mismo resultado que los cuadernos)
└── triaje_cli.py             # Triaje desde la terminal
```

## Instalación y uso

Requisitos: **Python 3.12** (fijado 3.12.7 en `.python-version`) y `git`.
No uses Python 3.13 o 3.14: `numpy==1.26.4` no se instala en esas versiones.

### Windows (PowerShell)

```powershell
git clone https://github.com/Diegootm/triaje-quejas-resenas-ecommerce.git
cd triaje-quejas-resenas-ecommerce
py -3.12 -m venv .venv             # si no tienes 3.12: py install 3.12
.\.venv\Scripts\Activate.ps1       # si PowerShell lo bloquea: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements.txt
python preparar_entorno.py         # 1. recursos de NLTK (una sola vez)
python entrenar_agente.py          # 2. entrena (≈22 s) y guarda modelos/
streamlit run app/app_streamlit.py # 3. interfaz en http://localhost:8501
```

### Linux / macOS

```bash
git clone https://github.com/Diegootm/triaje-quejas-resenas-ecommerce.git
cd triaje-quejas-resenas-ecommerce
python3.12 -m venv .venv           # o pyenv install 3.12.7 && pyenv local 3.12.7
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python preparar_entorno.py && python entrenar_agente.py
streamlit run app/app_streamlit.py
```

Otros comandos:

```bash
python triaje_cli.py "It runs very small, I had to return it for a larger size."   # triaje por terminal
python -m pytest -q                                                                  # pruebas (145)
```

## Número de grupos (K): lo elige el agente

`python entrenar_agente.py` ya no usa un K fijo:

1. **Barrido:** entrena K-means para K = 2..12 sobre TF-IDF + SVD (inercia, silueta sobre una
   muestra de 4 000 y tamaños de grupo).
2. **Regla:** descarta K < 3 y los K cuyo grupo menor tiene menos del 3 % del corpus; entre los
   restantes toma la mejor silueta `s*` y elige el **K más pequeño con silueta ≥ s\* − 15 % · |s\*|**.
3. **Resultado:** guarda el modelo con ese K (`modelo_kmeans.joblib`) y la decisión completa
   (tabla, descartados, umbral, margen, fecha) en `modelos/seleccion_k.json`.

`--k N` (2..12) **impone** un K manual; queda registrado como `origen: manual`. No existe `--barrido`:
el barrido siempre se hace.

Barrido real (semilla 42, 4 675 quejas):

| K | Inercia | Silueta | Grupo menor | Estado |
| --- | --- | --- | --- | --- |
| 2 | 944.68 | 0.0268 | 31.4 % | descartado: K menor que 3 |
| 3 | 931.45 | 0.0157 | 24.2 % | candidato |
| 4 | 920.60 | 0.0156 | 8.9 % | candidato |
| 5 | 911.66 | 0.0175 | 8.7 % | candidato |
| **6** | 903.62 | **0.0216** | 6.4 % | **elegido** |
| 7 | 896.25 | 0.0176 | 2.9 % | descartado: grupo menor < 3 % |
| 8 | 889.16 | 0.0184 | 2.995 % | descartado: grupo menor < 3 % |
| 9 | 881.04 | 0.0196 | 2.8 % | descartado: grupo menor < 3 % |
| 10 | 873.74 | 0.0192 | 3.3 % | candidato |
| 11 | 869.42 | 0.0249 | 3.2 % | candidato (mejor silueta, s\* = 0.02492) |
| 12 | 863.06 | 0.0030 | 2.4 % | descartado: grupo menor < 3 % |

Umbral = 0.02492 − 0.15 · 0.02492 = **0.02118**. K = 6 (0.02161) es el menor que lo alcanza, con un
margen de +0.00043 (≈ 2 % sobre el umbral).

### Interfaz

- La barra lateral muestra el K del agente y permite **explorar** otro K (2..12): solo se entrena
  K-means sobre el TF-IDF + SVD ya calculado; no se guarda en disco. Las quejas nuevas se asignan con
  el K activo y nunca reentrenan el modelo.
- **Selección de K:** la decisión del agente explicada, el barrido recalculable con gráficos y la
  comparación (ARI y contingencia) frente al modelo guardado.
- `triaje_cli.py` usa el K de `seleccion_k.json`.

### Nombres de los grupos

Cada grupo tiene un nombre **automático** (sus 3 términos TF-IDF principales) y, si existe, un
**nombre del informe**. Este se aplica solo si la **huella** de la partición coincide (SHA-256 de la
asignación de cada reseña): primero `modelos/nombres_grupos.json` (editable, con
`nombres_grupos_huella.json`) y luego los nombres del informe (`HUELLA_REFERENCIA` en
`codigo/etiquetado.py`, ya registrada para K = 6).

### Numeración de grupos

Al usuario se le muestran **desde 1** (Grupo 1 … Grupo 6). Internamente (modelos, JSON, huella) siguen
siendo 0-based. Equivalencia con el informe anterior, que numeraba desde 0: **grupo del informe + 1**.

| Informe | Interfaz | Quejas | Nombre del informe |
| --- | --- | --- | --- |
| 0 | Grupo 1 | 1 873 | Insatisfacción general de ajuste, color y tela |
| 1 | Grupo 2 | 435 | Calidad y precio |
| 2 | Grupo 3 | 361 | Expectativa vs. realidad |
| 3 | Grupo 4 | 300 | Silueta poco favorecedora |
| 4 | Grupo 5 | 720 | Tallaje (corre pequeña o grande) |
| 5 | Grupo 6 | 986 | Apariencia vs. modelo (forma holgada) |

## Pestaña "Cómo el agente crea las categorías"

Pulsa **Ver al agente crear las categorías**. Una línea de narración dice qué hace el agente en cada paso.
**Todo se calcula de verdad**; solo el ritmo de reproducción se puede frenar y la pantalla lo indica.

- **Fase 1 · Elegir K:** cada K = 2..12 se entrena en vivo y aparece una miniatura del mapa 2D coloreada
  por esa partición, mientras la curva de silueta se llena punto a punto. Los K descartados salen en gris
  con su motivo; al final se resalta el K elegido, se muestra la explicación y la comprobación contra
  `seleccion_k.json`.
- **Fase 2 · Formar los grupos:** mapa 2D grande. Las quejas parten grises; las 10 inicializaciones
  llenan una tira (inercia final, ARI frente al modelo, ganadora resaltada); se animan las semillas de
  k-means++ y las iteraciones de la ganadora, con estela de los centroides y contadores de iteración,
  inercia y quejas que cambiaron de grupo. Al terminar, el cursor muestra un fragmento de cada reseña.
- **Fase 3 · Nombrar:** cada grupo recibe su rótulo en el mapa, sus términos aparecen uno a uno y su
  tamaño sube contando, con el nombre automático y el del informe.
- **Velocidades:** *Velocidad real* (se espera lo que tardó cada cálculo), *Lenta* (0,5 s por paso),
  *Muy lenta* (1,2 s) y *Didáctica* (2,5 s). **Para la demostración se recomienda "Lenta"** (≈ 51 s en
  total). La primera pulsación calcula en vivo (≈ 11 s de cálculo y dibujo); las siguientes repiten la
  grabación sin recalcular. A velocidad real el dibujo es más lento que el cálculo.
- **Submuestra:** el mapa y las miniaturas dibujan una **muestra fija de 1 500 de las 4 675 quejas**
  (aleatoria, semilla 42) porque dibujar todas no era fluido (medido en Chrome). Los cálculos usan las
  quejas completas.
- **Avisos fijos:** el dibujo 2D explica solo ≈ 4.8 % de la varianza; su silueta (0.174) es la
  *separación del dibujo 2D*, no la calidad del modelo (silueta en 100 dimensiones: 0.0216); el modelo
  final es la mejor de 10 inicializaciones y la animación muestra a esa ganadora.

## Cuadernos de Jupyter (alternativa a `entrenar_agente.py`)

Los cuadernos usan el kernel llamado **`triaje312`**. Regístralo una vez con el entorno activado:

```bash
python -m ipykernel install --user --name triaje312 --display-name "Python 3 (venv)"
jupyter lab
```

Ejecuta en orden `01` → `02` → `03` → `04`. El cuaderno 03 guarda los modelos en `modelos/`.

## Resultados (reproducibles con `random_state = 42`)

| Indicador | Valor |
| --- | --- |
| Quejas seleccionadas | 4 675 de 22 641 (20.6 %); 465 solo por VADER |
| Matriz TF-IDF | 4 675 × 5 000, densidad 0.674 % |
| Varianza explicada por SVD(100) | 21.2 % |
| K elegido por el agente | 6 (regla de silueta con tolerancia del 15 %) |
| Inercia final | 903.62 (30 iteraciones; mejor de 10 inicializaciones) |
| Silueta | 0.0216 |
| ARI medio (10 corridas, n_init = 1) | 0.4014 ± 0.1107 |

## Limitaciones conocidas

- **Silueta baja y margen estrecho:** 0.0216 es muy baja y K = 6 supera el umbral por solo +0.00043;
  la regla es una heurística, no una prueba de calidad. K = 11 tiene mejor silueta pero grupos pequeños.
  Los grupos se mezclan y el Grupo 1 es amplio (40 %) y poco específico.
- **Vista 2D:** conserva ≈ 4.8 % de la varianza; en el dibujo los grupos se ven más mezclados de lo
  que están en 100 dimensiones.
- **La huella solo cubre las asignaciones** de reseñas a grupos, no los centroides ni el vectorizador.
  Un `nombres_grupos.json` antiguo (sin archivo de huella) se ignora: **vuelve a ejecutar
  `python entrenar_agente.py`**.
- **Pruebas:** las que guardan la secuencia de inicializaciones de Lloyd dependen de la versión de
  scikit-learn fijada (1.5.2) y fallan a propósito si cambia. La carga de un CSV no está cubierta por
  `AppTest` (no puede subir archivos); solo se prueba `clasificar_lote`.
- **`pytest -s`:** joblib/loky imprime en stderr una línea sobre núcleos físicos (`wmic` no existe en
  Windows 11). Es inofensiva.
- `too` no está protegido como negación/modificador; la lematización no usa categoría gramatical; no
  se normalizan los vectores tras el SVD; no hay partición de prueba separada.
- El modelo solo entiende inglés: un texto en español no comparte vocabulario y la app avisa.

## Notas de mantenimiento

- `modelos/` y los `*.joblib` no se versionan; se generan al entrenar.
- `pyarrow` está fijado en 17.0.0 porque las versiones nuevas exigen NumPy 2 y rompen Streamlit con
  `numpy==1.26.4`.
