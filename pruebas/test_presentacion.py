"""La numeración visible de los grupos parte de 1 y está en un solo lugar."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codigo import config, etiquetado  # noqa: E402
from codigo.presentacion import etiqueta_grupo, numero_grupo  # noqa: E402


def test_el_ayudante_mapea_0_a_1():
    assert config.PRIMER_GRUPO_VISIBLE == 1
    assert [numero_grupo(g) for g in range(3)] == [1, 2, 3]
    assert numero_grupo(np.int64(4)) == 5
    assert etiqueta_grupo(4) == "Grupo 5"


def test_los_nombres_automaticos_usan_la_numeracion_visible():
    nombres = etiquetado.etiquetar_clusters({0: ["a", "b", "c", "d"], 4: ["size", "small", "run"]})
    assert nombres == {0: "Grupo 1: a, b, c", 4: "Grupo 5: size, small, run"}


def test_la_huella_no_depende_de_la_numeracion_visible(monkeypatch):
    etiquetas = np.array([0, 0, 1, 2, 2, 1])
    antes = etiquetado.huella_particion(etiquetas)
    monkeypatch.setattr(config, "PRIMER_GRUPO_VISIBLE", 7)
    assert etiquetado.huella_particion(etiquetas) == antes


@pytest.mark.parametrize("grupo", [0, 5])
def test_los_ids_internos_siguen_desde_cero(grupo):
    nombres = etiquetado.NOMBRES_CLUSTER
    assert grupo in nombres and 6 not in nombres
