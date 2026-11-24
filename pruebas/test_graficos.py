"""Figuras: barras estáticas desde 0, mapa 2D, miniaturas, curva de silueta y tira de inicializaciones."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import graficos  # noqa: E402
from codigo import config  # noqa: E402

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


# --------------------------------------------------------------------------
# Mapa 2D, miniaturas, silueta y tira de inicializaciones
# --------------------------------------------------------------------------
PUNTOS = np.random.RandomState(0).normal(size=(60, 2))
ETIQ = np.arange(60) % 3


def test_la_configuracion_del_mapa_no_trae_zoom_ni_barra():
    assert graficos.CONFIG_MAPA == {"displayModeBar": False, "scrollZoom": False}


def test_mapa_sin_etiquetas_pone_todos_los_puntos_en_gris_y_sin_cursor():
    fig = graficos.figura_mapa(PUNTOS)
    assert fig.data[0].marker.color == graficos.COLOR_SIN_GRUPO and fig.data[0].hoverinfo == "skip"
    assert fig.layout.hovermode is False and fig.layout.dragmode is False
    assert len(fig.data[0].x) == 60 and fig.layout.height == config.MAPA_ALTO_PX


def test_mapa_colorea_por_grupo_con_una_escala_discreta():
    fig = graficos.figura_mapa(PUNTOS, ETIQ, k=3)
    marcador = fig.data[0].marker
    assert list(marcador.color) == list(ETIQ) and (marcador.cmin, marcador.cmax) == (-0.5, 2.5)
    colores = [c for _, c in marcador.colorscale]
    assert colores[0] == colores[1] == graficos.PALETA_GRUPOS[0] and colores[-1] == graficos.PALETA_GRUPOS[2]


def test_el_rango_de_los_ejes_no_depende_de_lo_que_se_dibuja():
    a = graficos.figura_mapa(PUNTOS)
    b = graficos.figura_mapa(PUNTOS, ETIQ, 3, centros=np.zeros((3, 2)))
    assert list(a.layout.xaxis.range) == list(b.layout.xaxis.range)
    assert list(a.layout.yaxis.range) == list(b.layout.yaxis.range)
    assert PUNTOS[:, 0].min() > a.layout.xaxis.range[0] and PUNTOS[:, 0].max() < a.layout.xaxis.range[1]


def test_centroides_con_estela_numerados_desde_1():
    centros = np.array([[0.0, 0.0], [1.0, 1.0]])
    fig = graficos.figura_mapa(PUNTOS, ETIQ, 3, centros=centros, estela=[centros - 0.5, centros - 0.2])
    lineas = [t for t in fig.data if t.mode == "lines"]
    marcas = [t for t in fig.data if t.mode == "markers+text"]
    assert len(lineas) == 2 and [list(t.x) for t in lineas][0] == [-0.5, -0.2, 0.0]    # estela + posición actual
    assert [t.text[0] for t in marcas] == ["1", "2"]


def test_rotulos_en_el_centroide_de_cada_grupo():
    centros = np.array([[0.0, 0.0], [2.0, 3.0]])
    fig = graficos.figura_mapa(PUNTOS, ETIQ, 3, centros=centros, rotulos={1: "Grupo 2: fit"})
    nota = fig.layout.annotations[0]
    assert (nota.x, nota.y, nota.text) == (2.0, 3.0, "Grupo 2: fit")


def test_el_cursor_solo_aparece_si_se_pasan_fragmentos_y_numera_desde_1():
    textos = [f"reseña {i}" for i in range(60)]
    fig = graficos.figura_mapa(PUNTOS, ETIQ, 3, fragmentos=textos)
    assert fig.data[0].hoverinfo == "text" and fig.layout.hovermode == "closest"
    assert fig.data[0].hovertext[0] == "<b>Grupo 1</b><br>reseña 0" and fig.data[0].hovertext[2].startswith("<b>Grupo 3</b>")
    gris = graficos.figura_mapa(PUNTOS, fragmentos=textos)
    assert gris.data[0].hovertext[0] == "reseña 0"


def test_miniatura_pequena_estatica_y_con_borde_solo_si_es_la_destacada():
    normal = graficos.figura_miniatura(PUNTOS, ETIQ, 3)
    elegida = graficos.figura_miniatura(PUNTOS, ETIQ, 3, destacada=True)
    assert normal.layout.height == config.MINIATURA_ALTO_PX and normal.layout.xaxis.visible is False
    assert normal.data[0].marker.size == config.TAMANO_PUNTO_MINIATURA
    assert not normal.layout.shapes and elegida.layout.shapes[0].line.color == graficos.COLOR_DESTACADO
    assert "K = 3" in normal.layout.annotations[0].text


def test_silueta_marca_descartados_candidatos_umbral_y_elegido():
    ks, s = [2, 3, 4, 5], [0.027, 0.016, 0.0156, 0.0175]
    antes = graficos.figura_silueta(ks, s, {2: "K menor que 3"})
    assert list(antes.data[0].marker.symbol) == ["x", "circle", "circle", "circle"] and not antes.layout.shapes
    despues = graficos.figura_silueta(ks, s, {2: "K menor que 3"}, umbral=0.017, k_elegido=5)
    assert list(despues.data[0].marker.color) == [graficos.COLOR_DESCARTADO, graficos.COLOR_BARRA, graficos.COLOR_BARRA,
                                                  graficos.COLOR_DESTACADO]
    assert despues.layout.shapes[0].y0 == pytest.approx(0.017)
    assert list(despues.data[0].text) == ["0.0270", "0.0160", "0.0156", "0.0175"]
    assert despues.layout.xaxis.fixedrange and despues.layout.yaxis.range[0] <= 0


def test_silueta_con_candidatos_resalta_los_que_superan_el_umbral():
    fig = graficos.figura_silueta([3, 4, 5], [0.0160, 0.0156, 0.0175], {}, umbral=0.0159, k_elegido=5)
    assert list(fig.data[0].marker.color) == [graficos.COLOR_CANDIDATO, graficos.COLOR_BARRA, graficos.COLOR_DESTACADO]


def test_tira_de_inicializaciones_resalta_la_ganadora_y_parte_de_cero():
    corridas = [{"indice": i, "inercia": 905.0 + i, "ari": 0.9 + i / 100, "iteraciones": 20} for i in range(3)]
    fig = graficos.figura_inicializaciones(corridas, indice_ganador=1)
    assert list(fig.data[0].marker.color) == [graficos.COLOR_BARRA, graficos.COLOR_DESTACADO, graficos.COLOR_BARRA]
    assert list(fig.data[0].y)[0] == "Inicialización 1" and fig.layout.xaxis.range[0] == 0
    assert list(fig.data[0].text) == ["905.00 · ARI 0.90", "906.00 · ARI 0.91", "907.00 · ARI 0.92"]
    assert fig.layout.height < 200
