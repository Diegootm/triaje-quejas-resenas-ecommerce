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
           ──► TF-IDF (1-2 gramas, 5 000 términos) ──► SVD truncada (100) ──► K-means (K = 6)
           ──► evaluación (silueta 0.0216, ARI 0.4014) ──► nombres post-hoc ──► interfaz de triaje
```

## Estructura

```text
.
├── app/app_streamlit.py      # Interfaz de triaje (conectada a los modelos)
├── codigo/
│   ├── preprocesamiento.py   # Carga, criterio de queja (VADER) y limpieza del texto
│   ├── caracteristicas.py    # TF-IDF
│   ├── agrupamiento.py       # SVD truncada, barrido de K y K-means
│   ├── evaluacion.py         # Silueta, ARI y contingencia
│   ├── etiquetado.py         # Términos top y nombres de grupos
│   └── triaje.py             # AgenteTriaje: sugiere el grupo de una queja nueva
├── cuadernos/                # 01 exploratorio · 02 preprocesamiento · 03 agrupamiento · 04 evaluación
├── datos/crudos/             # CSV de Kaggle (y archive.zip)
├── datos/procesados/         # resenas_filtradas.csv, resenas_limpias.csv
├── modelos/                  # vectorizador.joblib, svd_lsa.joblib, modelo_kmeans.joblib (se generan)
├── pruebas/test_proyecto.py  # pytest
├── entrenar_agente.py        # Entrena todo sin Jupyter (mismo resultado que los cuadernos)
└── triaje_cli.py             # Triaje desde la terminal
```

## Instalación

Requisitos: **Python 3.12** (fijado 3.12.7 en `.python-version`) y `git`.
No uses Python 3.13 o 3.14: `numpy==1.26.4` no se instala en esas versiones.

### Windows (PowerShell)

```powershell
git clone https://github.com/Diegootm/triaje-quejas-resenas-ecommerce.git
cd triaje-quejas-resenas-ecommerce
py -3.12 -m venv venv              # si no tienes 3.12: py install 3.12
.\venv\Scripts\Activate.ps1        # si PowerShell lo bloquea: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux / macOS

```bash
git clone https://github.com/Diegootm/triaje-quejas-resenas-ecommerce.git
cd triaje-quejas-resenas-ecommerce
python3.12 -m venv venv            # o pyenv install 3.12.7 && pyenv local 3.12.7
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Uso

```bash
python entrenar_agente.py                 # 1. entrena (≈40 s) y guarda los modelos en modelos/
streamlit run app/app_streamlit.py        # 2. abre la interfaz en http://localhost:8501
python triaje_cli.py "It runs very small, I had to return it for a larger size."   # alternativa por terminal
python -m pytest -q                       # pruebas (pruebas/test_proyecto.py y pruebas/test_seleccion_k.py)
```

`python entrenar_agente.py --barrido` imprime además la tabla de inercia y
silueta para K = 2..12 (método del codo) y una recomendación heurística de K.
`--k` acepta valores entre 2 y 12.

### Número de grupos (K) en la interfaz

En la barra lateral de Streamlit se elige K entre 2 y 12. El valor inicial es
el K del modelo guardado en `modelos/` (6 en la configuración del informe).

- Al cambiar K solo se entrena K-means sobre el TF-IDF + SVD ya calculado del
  corpus; TF-IDF y SVD no se reentrenan. Cada modelo queda en memoria mientras
  la app esté abierta y no se guarda en disco.
- Las quejas nuevas (individuales o por lote) se asignan con el K elegido y
  nunca reentrenan el modelo.
- La pestaña **Selección de K** calcula inercia y silueta para K = 2..12 (con
  barra de progreso), marca el K elegido, muestra una recomendación
  **heurística** y compara la partición con la del modelo guardado (ARI y
  tabla de contingencia).
- Los nombres de los grupos se atan a una partición concreta mediante su
  **huella** (SHA-256 de la asignación de cada reseña). Se usan, en este orden:
  1. `modelos/nombres_grupos.json`, si su huella (`modelos/nombres_grupos_huella.json`)
     coincide con la partición del modelo cargado (se pueden editar los nombres a mano);
  2. los nombres manuales del informe, si la partición tiene la huella registrada en
     `HUELLA_REFERENCIA` (`codigo/etiquetado.py`);
  3. en otro caso, los 3 términos de mayor peso TF-IDF de cada grupo.
- `HUELLA_REFERENCIA` está vacía hasta que el equipo la registre: entrenar con
  K = 6, verificar que el modelo reproduce el informe y copiar la huella que
  imprime `entrenar_agente.py`.
- Un `nombres_grupos.json` antiguo, sin archivo de huella, se ignora: vuelve a
  entrenar para generarlo.

### Cuadernos de Jupyter (alternativa a `entrenar_agente.py`)

Los cuadernos usan el kernel llamado **`triaje312`**. Regístralo una vez con el
entorno activado:

```bash
python -m ipykernel install --user --name triaje312 --display-name "Python 3 (venv)"
jupyter lab
```

Ejecuta en orden `01` → `02` → `03` → `04` (*Run → Run All Cells*). El cuaderno
03 guarda los modelos en `modelos/`.

## Resultados (reproducibles con `random_state = 42`)

| Indicador | Valor |
| --- | --- |
| Quejas seleccionadas | 4 675 de 22 641 (20.6 %); 465 solo por VADER |
| Matriz TF-IDF | 4 675 × 5 000, densidad 0.674 % |
| Varianza explicada por SVD(100) | 21.2 % |
| K elegido | 6 (codo + silueta + interpretabilidad) |
| Inercia final | 903.62 (30 iteraciones) |
| Silueta | 0.0216 |
| ARI medio (10 corridas, n_init = 1) | 0.4014 ± 0.1107 |

| Grupo | Quejas | Nombre (post-hoc) |
| --- | --- | --- |
| 0 | 1 873 | Insatisfacción general de ajuste, color y tela |
| 1 | 435 | Calidad y precio |
| 2 | 361 | Expectativa vs. realidad |
| 3 | 300 | Silueta poco favorecedora |
| 4 | 720 | Tallaje (corre pequeña o grande) |
| 5 | 986 | Apariencia vs. modelo (forma holgada) |

## Limitaciones conocidas (trabajo futuro)

- Silueta baja y estabilidad moderada: los temas de queja se mezclan.
- El grupo 0 es amplio (40 %) y poco específico.
- `too` no está protegido como negación/modificador ("too small" → "small").
- La lematización no usa categoría gramatical (`ran`, `wanted`, `looked` no se reducen).
- No se normalizan los vectores después del SVD.
- No hay partición de prueba separada.
- El modelo solo entiende inglés: un texto en español no comparte vocabulario
  y la app avisa en lugar de sugerir un grupo.

## Notas de mantenimiento

- `modelos/` y los `*.joblib` no se versionan; se generan al entrenar.
- `pyarrow` está fijado en 17.0.0 porque las versiones nuevas exigen NumPy 2 y
  rompen las tablas y gráficos de Streamlit con `numpy==1.26.4`.
