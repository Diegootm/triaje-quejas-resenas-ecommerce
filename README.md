# Triaje de quejas de reseñas de e-commerce

Base de un proyecto de análisis y agrupamiento (clustering) de reseñas de
clientes de una tienda de ropa online. El objetivo es identificar y priorizar
**quejas** dentro de un volumen grande de reseñas, agrupando los textos por
temas similares para que el equipo pueda revisarlos y atenderlos de forma
ordenada.

El dataset de trabajo es **Women's E-Commerce Clothing Reviews** (Kaggle,
textos en inglés). El texto se conserva en su idioma original: no se traduce,
porque la traducción puede alterar matices y dificultar la interpretación de
los términos que definen cada clúster.

## Objetivo

1. Preparar y limpiar las reseñas.
2. Representar el texto con TF-IDF.
3. Agrupar las reseñas con KMeans y elegir el número de clústeres.
4. Evaluar la calidad y la estabilidad del agrupamiento.
5. Etiquetar cada clúster con sus términos más representativos.
6. Exponer una interfaz mínima para revisar el resultado.

Esta etapa entrega la **estructura del repositorio, la documentación y
esqueletos de código**. Todavía no se han ejecutado entrenamientos ni pruebas,
y los módulos de `codigo/` lanzan `NotImplementedError` de forma intencional.

## Estructura del proyecto

```text
.
├── .gitignore
├── .python-version          # 3.12.7
├── README.md
├── requirements.txt
├── app/
│   └── app_streamlit.py     # Punto de entrada de la interfaz
├── codigo/
│   ├── __init__.py
│   ├── preprocesamiento.py  # Limpieza, stopwords, lematización
│   ├── caracteristicas.py   # Vectorización TF-IDF
│   ├── agrupamiento.py      # KMeans y selección de k
│   ├── evaluacion.py        # Silueta, ARI, estabilidad
│   └── etiquetado.py        # Términos top por clúster
├── cuadernos/
│   ├── 01_analisis_exploratorio.ipynb
│   ├── 02_preprocesamiento.ipynb
│   ├── 03_agrupamiento.ipynb
│   └── 04_evaluacion.ipynb
├── datos/
│   ├── crudos/              # archive.zip y CSV/XLSX de Kaggle (no versionados)
│   └── procesados/          # Salidas intermedias (no versionadas)
├── interfaz/
│   └── README.md            # Apunta a app/app_streamlit.py
├── modelos/                 # Modelos y vectorizadores serializados
└── pruebas/                 # Pruebas automatizadas
```


## Requisitos

- Linux (probado en una distribución de escritorio estándar).
- **Python 3.12.7** (ver `.python-version`).
- [pyenv](https://github.com/pyenv/pyenv) recomendado para fijar la versión.
- `git`, `unzip` y acceso a los datos de Kaggle.

## Instalación paso a paso

### 1. Clonar el repositorio y entrar en él

```bash
git clone <URL_DEL_REPOSITORIO>
cd triaje-quejas-resenas-ecommerce
```

### 2. Instalar y fijar Python 3.12.7 con pyenv

```bash
# Si pyenv todavía no está instalado, sigue las instrucciones oficiales.
pyenv install 3.12.7
pyenv local 3.12.7        # lee .python-version
python --version          # debe mostrar Python 3.12.7
```

Si no usas pyenv, asegúrate de que `python3.12` esté disponible en el sistema.

### 3. Crear y activar el entorno virtual

```bash
python -m venv venv
source venv/bin/activate
```

### 4. Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Las versiones están fijadas con `==` para facilitar la reproducibilidad.

### 5. Registrar el kernel de Jupyter (opcional, recomendado)

```bash
python -m ipykernel install --user --name triaje-quejas --display-name "Python (triaje-quejas)"
```

## Preparar los datos

1. Descarga el dataset **Women's E-Commerce Clothing Reviews** desde Kaggle.
2. Coloca el archivo en `datos/crudos/`. El repositorio ya contiene
   `datos/crudos/archive.zip`.
3. Extrae el contenido dentro de `datos/crudos/` (sin modificar el zip
   original):

   ```bash
   unzip datos/crudos/archive.zip -d datos/crudos/
   ```

   Normalmente se obtiene un CSV como
   `Womens Clothing E-Commerce Reviews.csv`.

4. Verifica las columnas esperadas:

   ```text
   Clothing ID, Age, Title, Review Text, Rating, Recommended IND,
   Positive Feedback Count, Division Name, Department Name, Class Name
   ```

5. Genera los datos procesados con los módulos de `codigo/` y los cuadernos.
   Por ahora esa lógica está pendiente: las salidas deben escribirse en
   `datos/procesados/`, que no se versiona.

Ejemplo de comprobación rápida del CSV (cuando esté implementada la carga):

```python
import pandas as pd

df = pd.read_csv("datos/crudos/Womens Clothing E-Commerce Reviews.csv")
print(df.shape)
print(df.columns.tolist())
```

## Ejecutar los cuadernos

```bash
jupyter lab
```

Abre los cuadernos de `cuadernos/` en orden:

1. `01_analisis_exploratorio.ipynb`
2. `02_preprocesamiento.ipynb`
3. `03_agrupamiento.ipynb`
4. `04_evaluacion.ipynb`

Selecciona el kernel `Python (triaje-quejas)` o el del entorno `venv`.

## Ejecutar la interfaz

```bash
streamlit run app/app_streamlit.py
```

La app es un esqueleto funcional: muestra la pantalla, permite escribir una
reseña o cargar un CSV, pero **no clasifica todavía** porque no hay un modelo
entrenado. Cuando exista un artefacto en `modelos/`, se conectará al pipeline
de preprocesamiento + TF-IDF + KMeans.

## Definición propuesta de "queja"

El dataset no incluye una columna explícita de queja, así que hay que
definirla. Las dos opciones principales son:

| Opción | Criterio | Ventajas | Limitaciones |
| --- | --- | --- | --- |
| Por `Rating` | Considerar queja una reseña con `Rating <= 2` (o `<= 3`) | Simple, objetiva y fácil de reproducir | Ignora reseñas negativas con rating alto y reseñas positivas con rating bajo |
| Por `Recommended IND` | Considerar queja `Recommended IND == 0` | Refleja la recomendación real del cliente | Es binaria y puede no coincidir con la intensidad del problema |

También puede combinarse: por ejemplo, `Rating <= 3` **o**
`Recommended IND == 0`, y revisar los casos discrepantes.

**Decisión pendiente de documentar.** La definición elegida se fijará antes de
entrenar y se justificará en los cuadernos y en este README. Mientras tanto,
el pipeline debe conservar ambas columnas (`Rating` y `Recommended IND`) sin
modificarlas.

## Datos y modelos en el repositorio

- No se versionan los CSV/XLSX crudos ni los datos procesados.
- El `archive.zip` existente se mantiene y no se ignora.
- No se versionan los modelos serializados (`modelos/`), salvo que en el futuro
  se acuerde incluir un artefacto pequeño y necesario para la demo.
- Los `.gitkeep` conservan las carpetas vacías.
- `.env` y variantes están ignorados; no se almacenan secretos.

## Estado y validación

- Estructura del repositorio: creada.
- Dependencias: declaradas y fijadas en `requirements.txt`.
- Módulos de `codigo/`: esqueletos documentados, sin lógica implementada.
- Cuadernos: vacíos, pendientes de desarrollo.
- Entrenamientos: no ejecutados.
- Pruebas: no ejecutadas; `pytest` está incluido para la siguiente etapa.
- Interfaz: stub funcional sin modelo.

Validaciones realizadas en esta etapa: revisión de la estructura de carpetas y
comprobación de sintaxis (AST) de los módulos de `codigo/` y de
`app/app_streamlit.py`. No se instalaron dependencias ni se ejecutó la app.

## Próximos pasos

1. Implementar `codigo/preprocesamiento.py` y validar con una muestra.
2. Implementar TF-IDF y KMeans, y elegir `k` con silueta.
3. Evaluar estabilidad y etiquetar clústeres.
4. Documentar la definición de queja adoptada.
5. Conectar el modelo a `app/app_streamlit.py`.
6. Agregar pruebas en `pruebas/`.
