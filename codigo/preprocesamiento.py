"""Selección de quejas y preprocesamiento de reseñas (cuadernos 01 y 02).

Responsabilidades:

1. Cargar el dataset crudo (CSV de Kaggle o ``archive.zip``).
2. Marcar las reseñas que son queja combinando tres señales
   (``Rating <= 2`` o ``Recommended IND == 0`` o VADER ``compound <= -0.05``).
3. Unir ``Title`` + ``Review Text``.
4. Minúsculas, expansión de contracciones, limpieza, tokenización,
   eliminación de palabras vacías conservando negaciones y lematización.

La lógica es idéntica a la de los cuadernos 01 y 02, de modo que el texto
limpio que produce este módulo es el mismo que se usó para entrenar.

Nota: el texto NO se traduce; se mantiene en inglés (idioma del corpus).
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Iterable, Sequence

import nltk
import pandas as pd

# Columnas de texto del dataset Women's E-Commerce Clothing Reviews.
COLUMNAS_TEXTO: tuple[str, ...] = ("Title", "Review Text")

# Umbral estándar de VADER para considerar negativo un texto (Hutto y Gilbert, 2014).
UMBRAL_VADER = -0.05

# Recurso de NLTK -> ruta que ``nltk.data.find`` debe encontrar (fuente única).
RECURSOS_NLTK: dict[str, str] = {
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4",
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "vader_lexicon": "sentiment/vader_lexicon",
}

COMANDO_PREPARAR_ENTORNO = "python preparar_entorno.py"

# Contracciones frecuentes en inglés (prioriza las que contienen negación).
CONTRACCIONES = {
    "didn't": "did not", "doesn't": "does not", "don't": "do not",
    "wasn't": "was not", "weren't": "were not", "isn't": "is not", "aren't": "are not",
    "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
    "won't": "will not", "wouldn't": "would not", "can't": "can not", "cannot": "can not",
    "couldn't": "could not", "shouldn't": "should not", "mustn't": "must not",
    "needn't": "need not", "mightn't": "might not", "shan't": "shall not",
    "it's": "it is", "that's": "that is", "there's": "there is", "here's": "here is",
    "i'm": "i am", "i've": "i have", "i'd": "i would", "i'll": "i will",
    "you're": "you are", "we're": "we are", "they're": "they are",
    "he's": "he is", "she's": "she is", "what's": "what is", "let's": "let us",
}

# Negaciones que SIEMPRE se conservan aunque estén en la lista de palabras vacías.
NEGACIONES = frozenset({
    "not", "no", "never", "without", "nor", "none", "neither", "cannot",
    # formas que NLTK separa de contracciones si no se expandieron antes:
    "don", "didn", "doesn", "wasn", "weren", "isn", "aren", "hasn", "haven",
    "hadn", "won", "wouldn", "shouldn", "couldn", "mustn", "needn", "mightn", "shan", "ain",
})

_patron_contr = re.compile(r"\b(" + "|".join(map(re.escape, CONTRACCIONES)) + r")\b")
_patron_limpieza = re.compile(r"[^a-z\s]")

_stopwords_en: frozenset[str] | None = None
_lematizador = None
_recursos_verificados = False  # evita consultar el disco en cada texto


class RecursoNltkFaltante(LookupError):
    """Faltan recursos de NLTK; el mensaje indica cómo instalarlos."""


def recursos_nltk_faltantes() -> list[str]:
    """Nombres de los recursos de NLTK que no están instalados (no descarga nada)."""
    return [nombre for nombre, ruta in RECURSOS_NLTK.items() if not _recurso_instalado(ruta)]


def _recurso_instalado(ruta: str) -> bool:
    """Un recurso puede estar descomprimido o como .zip (NLTK carga ambos)."""
    for candidata in (ruta, ruta + ".zip"):
        try:
            nltk.data.find(candidata)
            return True
        except LookupError:
            continue
    return False


def verificar_recursos_nltk() -> None:
    """Falla con un mensaje claro si falta algún recurso de NLTK (sin descargas)."""
    global _recursos_verificados
    if _recursos_verificados:
        return
    faltantes = recursos_nltk_faltantes()
    if faltantes:
        raise RecursoNltkFaltante(
            f"Faltan recursos de NLTK: {', '.join(faltantes)}. "
            f"Instálalos una vez con: {COMANDO_PREPARAR_ENTORNO}"
        )
    _recursos_verificados = True


def _stopwords() -> frozenset[str]:
    global _stopwords_en
    if _stopwords_en is None:
        from nltk.corpus import stopwords

        verificar_recursos_nltk()
        _stopwords_en = frozenset(stopwords.words("english")) - NEGACIONES
    return _stopwords_en


def _lemmatizer():
    global _lematizador
    if _lematizador is None:
        from nltk.stem import WordNetLemmatizer

        _lematizador = WordNetLemmatizer()
    return _lematizador


# --------------------------------------------------------------------------
# Carga y selección de quejas (cuaderno 01)
# --------------------------------------------------------------------------
def cargar_resenas(ruta: str | Path, **kwargs) -> pd.DataFrame:
    """Carga el CSV de reseñas (o el ``archive.zip`` de Kaggle).

    Descarta las columnas de índice sin nombre (``Unnamed: 0``) y las reseñas
    sin ``Review Text``, porque TF-IDF y VADER necesitan texto.
    """
    ruta = Path(ruta)
    if ruta.suffix == ".zip":
        with zipfile.ZipFile(ruta) as zf:
            nombre_csv = next(n for n in zf.namelist() if n.endswith(".csv"))
            with zf.open(nombre_csv) as f:
                df = pd.read_csv(f, **kwargs)
    else:
        df = pd.read_csv(ruta, **kwargs)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    return df.dropna(subset=["Review Text"]).reset_index(drop=True)


def marcar_quejas(df: pd.DataFrame, umbral_vader: float = UMBRAL_VADER) -> pd.DataFrame:
    """Agrega las señales de queja y la columna ``es_queja`` (Fórmula 22 del informe).

    queja = Rating <= 2  OR  Recommended IND == 0  OR  VADER compound <= -0.05
    """
    from nltk.sentiment import SentimentIntensityAnalyzer

    verificar_recursos_nltk()
    sia = SentimentIntensityAnalyzer()

    df = df.copy()
    df["vader_compound"] = df["Review Text"].astype(str).apply(lambda t: sia.polarity_scores(t)["compound"])
    df["calif_baja"] = df["Rating"] <= 2
    df["no_recomienda"] = df["Recommended IND"] == 0
    df["sent_negativo"] = df["vader_compound"] <= umbral_vader
    df["es_queja"] = df["calif_baja"] | df["no_recomienda"] | df["sent_negativo"]
    return df


# --------------------------------------------------------------------------
# Preprocesamiento del texto (cuaderno 02)
# --------------------------------------------------------------------------
def unir_texto(df: pd.DataFrame, columnas: Sequence[str] = COLUMNAS_TEXTO) -> pd.Series:
    """Concatena ``Title`` + ``Review Text`` (nulos como cadena vacía)."""
    serie = df[columnas[0]].fillna("")
    for col in columnas[1:]:
        serie = serie + " " + df[col].fillna("")
    return serie.str.strip()


def expandir_contracciones(texto: str) -> str:
    """``didn't`` -> ``did not`` (se hace ANTES de quitar el apóstrofo)."""
    return _patron_contr.sub(lambda m: CONTRACCIONES[m.group(0)], texto)


def limpiar_texto(texto: str) -> str:
    """Minúsculas + expansión de contracciones + quitar todo lo que no sea a-z o espacio."""
    texto = str(texto).lower()
    texto = expandir_contracciones(texto)
    return _patron_limpieza.sub(" ", texto)


def tokenizar(texto: str) -> list[str]:
    """Separa el texto limpio en tokens con ``word_tokenize`` de NLTK."""
    from nltk.tokenize import word_tokenize

    verificar_recursos_nltk()
    return word_tokenize(texto)


def eliminar_stopwords(tokens: Iterable[str], stopwords: Iterable[str] | None = None) -> list[str]:
    """Quita palabras vacías y tokens de <= 2 letras, conservando SIEMPRE las negaciones."""
    vacias = _stopwords() if stopwords is None else frozenset(stopwords) - NEGACIONES
    return [t for t in tokens if (t not in vacias and len(t) > 2) or t in NEGACIONES]


def lematizar(tokens: Iterable[str]) -> list[str]:
    """Lematiza cada token con ``WordNetLemmatizer`` (categoría por defecto: sustantivo)."""
    verificar_recursos_nltk()
    lem = _lemmatizer()
    return [lem.lemmatize(t) for t in tokens]


def preprocesar(texto: str) -> str:
    """Pipeline completo para un texto: limpiar -> tokenizar -> stopwords -> lematizar."""
    return " ".join(lematizar(eliminar_stopwords(tokenizar(limpiar_texto(texto)))))


def preprocesar_dataframe(
    df: pd.DataFrame,
    columna_texto: str = "texto",
    columna_salida: str = "texto_limpio",
) -> pd.DataFrame:
    """Une título y reseña, aplica :func:`preprocesar` y quita filas que queden vacías."""
    df = df.copy()
    if columna_texto not in df.columns:
        df[columna_texto] = unir_texto(df)
    df[columna_salida] = df[columna_texto].apply(preprocesar)
    return df[df[columna_salida].str.strip().str.len() > 0].reset_index(drop=True)
