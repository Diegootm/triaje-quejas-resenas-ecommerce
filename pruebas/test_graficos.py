"""Figuras de las barras: estáticas, con el eje desde 0 y con los datos que muestran."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import graficos  # noqa: E402

ETIQUETAS = ["Grupo 1: not, fit, color", "Grupo 2: quality, price, poor", "Grupo 3: wanted, love"]
DISTANCIAS = [0.5388, 0.4520, 0.6099]


@pytest.fixture(params=["distancias", "distribucion"])
def figura(request):
    if request.param == "distancias":
        return graficos.figura_distancias(ETIQUETAS, DISTANCIAS, indice_sugerido=1)
    return graficos.figura_distribucion(ETIQUETAS, [1873, 435, 361])


def test_la_configuracion_es_estatica():
    assert graficos.CONFIG_ESTATICO == {"staticPlot": True, "displayModeBar": False}


def test_sin_zoom_ni_arrastre(figura):
    assert figura.layout.dragmode is False
    assert figura.layout.xaxis.fixedrange is True and figura.layout.yaxis.fixedrange is True


def test_el_eje_empieza_en_cero_y_no_hay_valores_negativos(figura):
    inicio, fin = figura.layout.xaxis.range
    assert inicio == 0
    barra = figura.data[0]
    assert min(barra.x) >= 0 and fin == pytest.approx(max(barra.x) * graficos.MARGEN_EJE)
    assert figura.layout.xaxis.rangemode == "nonnegative"


def test_distancias_conservan_los_datos_y_destacan_el_grupo_sugerido():
    fig = graficos.figura_distancias(ETIQUETAS, DISTANCIAS, indice_sugerido=1)
    barra = fig.data[0]
    assert list(barra.y) == ETIQUETAS and list(barra.x) == DISTANCIAS and barra.orientation == "h"
    assert list(barra.marker.color) == [graficos.COLOR_BARRA, graficos.COLOR_DESTACADO, graficos.COLOR_BARRA]
    assert list(barra.text) == ["0.539", "0.452", "0.610"]
    assert fig.layout.yaxis.autorange == "reversed"      # el grupo 1 arriba


def test_distribucion_conserva_los_datos():
    fig = graficos.figura_distribucion(ETIQUETAS, [1873, 435, 361])
    assert list(fig.data[0].x) == [1873, 435, 361] and list(fig.data[0].text) == ["1,873", "435", "361"]


def test_valores_en_cero_no_rompen_el_rango():
    fig = graficos.figura_distancias(["a", "b"], [0.0, 0.0], 0)
    assert list(fig.layout.xaxis.range) == [0, 1.0]
