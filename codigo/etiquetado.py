"""Interpretación de los grupos: términos representativos y nombres (cuaderno 04).

Los nombres de :data:`NOMBRES_CLUSTER` son una interpretación humana posterior
(post-hoc) del modelo de referencia (K=6, semilla 42). NO se usan para
entrenar: K-means solo ve la matriz TF-IDF/SVD.

Los nombres se atan a una partición concreta mediante su huella
(:func:`huella_particion`). Si la partición no coincide con la de los nombres
manuales o persistidos, se usan etiquetas derivadas de los términos.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from codigo.config import ARCHIVO_HUELLA_NOMBRES, ARCHIVO_NOMBRES, N_TERMINOS_TOP

NOMBRES_CLUSTER: dict[int, str] = {
    0: "Insatisfacción general de ajuste, color y tela",
    1: "Calidad y precio (tela pobre, no vale lo que cuesta)",
    2: "Expectativa vs. realidad ('quería que me gustara')",
    3: "Silueta poco favorecedora (not flattering)",
    4: "Tallaje (la talla corre pequeña o grande)",
    5: "Apariencia vs. modelo (forma holgada / boxy)",
}

# Huella (ver huella_particion) de la partición del modelo de referencia sobre
# datos/procesados/resenas_limpias.csv. Los nombres de NOMBRES_CLUSTER solo se
# aplican (como nombre secundario) a una partición con esta huella exacta.
#
# Registrada tras comprobar que el modelo K = 6 entrenado por el agente reproduce el
# informe: tamaños de grupo, distancias de la Tabla 44, inercia, silueta y los
# términos principales de cada grupo (pruebas/test_nombres.py). Para registrar otra,
# entrenar, comprobar esas mismas cifras y copiar la huella que imprime
# entrenar_agente.py. Si vale None, NOMBRES_CLUSTER no se aplica a ningún modelo.
HUELLA_REFERENCIA: str | None = "2921bd90a2bf24facec9d4b51dce5ee170c702798f7bc50eb25ceea440722ca7"


def huella_particion(etiquetas: Sequence[int]) -> str:
    """SHA-256 de la asignación completa (reseña por reseña, en el orden del corpus).

    Dos particiones con los mismos tamaños de grupo pero distinta asignación
    tienen huellas distintas. También cambia si cambia el corpus o su orden.
    """
    etiquetas = np.asarray(etiquetas, dtype="<i8")
    k = int(etiquetas.max()) + 1 if etiquetas.size else 0
    h = hashlib.sha256(f"particion-v1|{etiquetas.size}|{k}|".encode("ascii"))
    h.update(etiquetas.tobytes())
    return h.hexdigest()


def es_modelo_referencia(etiquetas: Sequence[int], huella_referencia: str | None = HUELLA_REFERENCIA) -> bool:
    """True solo si la partición es idéntica a la de referencia (misma huella).

    Tener K = 6 o los mismos tamaños de grupo no basta: los IDs de K-means son
    arbitrarios y la misma distribución de tamaños puede venir de otra asignación.
    """
    return huella_referencia is not None and huella_particion(etiquetas) == huella_referencia


def nombres_manuales_validos(
    etiquetas: Sequence[int],
    nombres: Mapping[int, str] = NOMBRES_CLUSTER,
    huella_referencia: str | None = HUELLA_REFERENCIA,
) -> dict[int, str]:
    """Devuelve ``nombres`` si la partición es la de referencia; si no, un diccionario vacío."""
    return dict(nombres) if es_modelo_referencia(etiquetas, huella_referencia) else {}


@dataclass(frozen=True)
class NombresPersistidos:
    """Contenido de ``nombres_grupos.json`` y de su huella (``None`` si el archivo es del formato anterior)."""

    nombres: dict[int, str]
    huella: str | None


def guardar_nombres(nombres: Mapping[int, str], etiquetas: Sequence[int], directorio: str | Path) -> None:
    """Guarda ``nombres_grupos.json`` (mismo formato de siempre) y la huella de la partición."""
    directorio = Path(directorio)
    directorio.mkdir(parents=True, exist_ok=True)
    etiquetas = np.asarray(etiquetas)
    (directorio / ARCHIVO_NOMBRES).write_text(
        json.dumps({str(int(k)): v for k, v in nombres.items()}, ensure_ascii=False, indent=2), encoding="utf-8")
    meta = {
        "huella_particion": huella_particion(etiquetas),
        "k": int(etiquetas.max()) + 1 if etiquetas.size else 0,
        "n_resenas": int(etiquetas.size),
    }
    (directorio / ARCHIVO_HUELLA_NOMBRES).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def cargar_nombres(directorio: str | Path) -> NombresPersistidos | None:
    """Lee los nombres persistidos. ``None`` si no existe ``nombres_grupos.json``.

    Lanza ``ValueError`` si alguno de los archivos tiene un formato inválido.
    """
    directorio = Path(directorio)
    ruta = directorio / ARCHIVO_NOMBRES
    if not ruta.exists():
        return None
    try:
        crudo = json.loads(ruta.read_text(encoding="utf-8"))
        nombres = {int(k): str(v) for k, v in crudo.items()}
    except (json.JSONDecodeError, AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"{ARCHIVO_NOMBRES} no tiene el formato {{\"0\": \"nombre\", ...}}: {error}") from error
    ruta_huella = directorio / ARCHIVO_HUELLA_NOMBRES
    huella = None
    if ruta_huella.exists():
        try:
            huella = str(json.loads(ruta_huella.read_text(encoding="utf-8"))["huella_particion"])
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError(f"{ARCHIVO_HUELLA_NOMBRES} no es válido: {error}") from error
    return NombresPersistidos(nombres, huella)


def elegir_nombres_manuales(
    etiquetas: Sequence[int],
    persistidos: NombresPersistidos | None = None,
    huella_referencia: str | None = HUELLA_REFERENCIA,
) -> tuple[dict[int, str], str, str | None]:
    """Decide qué nombres no automáticos corresponden a ``etiquetas``.

    Devuelve ``(nombres, origen, aviso)`` con ``origen`` en:

    - ``"archivo"``: ``nombres_grupos.json`` cuya huella coincide con la partición.
    - ``"referencia"``: :data:`NOMBRES_CLUSTER`, si la partición es la de referencia.
    - ``"terminos"``: ninguno corresponde; se usarán etiquetas derivadas de los términos.

    ``aviso`` explica por qué se ignoró ``nombres_grupos.json``, si fue el caso.
    """
    etiquetas = np.asarray(etiquetas)
    k = int(etiquetas.max()) + 1 if etiquetas.size else 0
    aviso = None
    if persistidos is not None:
        if persistidos.huella is None:
            aviso = (f"Se ignoró {ARCHIVO_NOMBRES}: no tiene {ARCHIVO_HUELLA_NOMBRES} (formato anterior), "
                     "así que no se puede comprobar a qué partición pertenece. Vuelve a entrenar para generarlo.")
        elif persistidos.huella != huella_particion(etiquetas):
            aviso = f"Se ignoró {ARCHIVO_NOMBRES}: pertenece a otra partición que la del modelo cargado."
        elif not set(persistidos.nombres) <= set(range(k)):
            aviso = f"Se ignoró {ARCHIVO_NOMBRES}: tiene IDs de grupo fuera de 0..{k - 1}."
        else:
            return dict(persistidos.nombres), "archivo", None
    if es_modelo_referencia(etiquetas, huella_referencia):
        return dict(NOMBRES_CLUSTER), "referencia", aviso
    return {}, "terminos", aviso


def terminos_top_por_cluster(
    matriz: sp.csr_matrix,
    etiquetas: Sequence[int],
    vectorizador: TfidfVectorizer,
    n_terminos: int = N_TERMINOS_TOP,
) -> dict[int, list[str]]:
    """Términos con mayor peso TF-IDF promedio dentro de cada grupo."""
    etiquetas = np.asarray(etiquetas)
    vocab = np.array(vectorizador.get_feature_names_out())
    resultado = {}
    for c in np.unique(etiquetas):
        medio = np.asarray(matriz[etiquetas == c].mean(axis=0)).ravel()
        resultado[int(c)] = list(vocab[medio.argsort()[::-1][:n_terminos]])
    return resultado


def etiquetar_clusters(
    terminos: Mapping[int, Sequence[str]],
    nombres_manuales: Mapping[int, str] | None = None,
) -> dict[int, str]:
    """Nombre manual si existe; si no, una etiqueta provisional con los 3 primeros términos."""
    nombres_manuales = nombres_manuales or {}
    return {
        c: nombres_manuales.get(c) or f"Grupo {c}: " + ", ".join(list(terms)[:3])
        for c, terms in terminos.items()
    }
