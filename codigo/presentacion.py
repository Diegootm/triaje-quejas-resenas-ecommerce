"""Cómo se muestran los grupos al usuario (única fuente de la numeración visible).

Internamente los grupos son 0, 1, ..., K-1 (modelos, huella, JSON). Al usuario se le
muestran numerados desde ``PRIMER_GRUPO_VISIBLE``. Todo lugar que muestre un número de
grupo debe pasar por :func:`numero_grupo`.
"""

from __future__ import annotations

from codigo.config import PRIMER_GRUPO_VISIBLE


def numero_grupo(grupo: int) -> int:
    """Número de grupo que ve el usuario (0 -> 1)."""
    return int(grupo) + PRIMER_GRUPO_VISIBLE


def etiqueta_grupo(grupo: int) -> str:
    """``"Grupo 5"`` para el grupo interno 4."""
    return f"Grupo {numero_grupo(grupo)}"
