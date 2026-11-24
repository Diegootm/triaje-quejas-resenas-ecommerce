"""Figuras de Plotly como funciones puras (sin Streamlit): reciben datos y devuelven ``go.Figure``.

Los gráficos de barras son ESTÁTICOS: sin zoom, desplazamiento, arrastre ni barra de
herramientas. Al mostrarlos hay que pasar ``config=CONFIG_ESTATICO``.
"""

from __future__ import annotations

from typing import Sequence

import plotly.graph_objects as go

CONFIG_ESTATICO = {"staticPlot": True, "displayModeBar": False}

COLOR_BARRA = "#9DB4CE"
COLOR_DESTACADO = "#E45756"
MARGEN_EJE = 1.1            # el eje X llega hasta el máximo por este factor (y siempre empieza en 0)
ALTO_MINIMO = 240
ALTO_POR_BARRA = 28


def _barras_horizontales(
    etiquetas: Sequence[str], valores: Sequence[float], titulo_x: str,
    colores: Sequence[str], formato_texto: str,
) -> go.Figure:
    """Barras horizontales estáticas; el eje X va de 0 a ``max · MARGEN_EJE`` (sin ticks negativos)."""
    maximo = max((float(v) for v in valores), default=0.0)
    fig = go.Figure(go.Bar(
        x=list(valores), y=list(etiquetas), orientation="h", marker_color=list(colores),
        text=[format(float(v), formato_texto) for v in valores], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x}<extra></extra>",
    ))
    fig.update_xaxes(range=[0, maximo * MARGEN_EJE if maximo > 0 else 1.0], fixedrange=True, title=titulo_x,
                     rangemode="nonnegative", showgrid=True)
    fig.update_yaxes(autorange="reversed", fixedrange=True, automargin=True)   # el primer grupo arriba
    fig.update_layout(
        dragmode=False, showlegend=False, bargap=0.25,
        height=max(ALTO_MINIMO, ALTO_POR_BARRA * len(etiquetas) + 80),
        margin=dict(l=10, r=40, t=10, b=40),
    )
    return fig


def figura_distancias(etiquetas: Sequence[str], distancias: Sequence[float], indice_sugerido: int) -> go.Figure:
    """Distancia de una queja a cada centroide (menor = más parecido); el grupo sugerido va destacado."""
    colores = [COLOR_DESTACADO if i == indice_sugerido else COLOR_BARRA for i in range(len(distancias))]
    return _barras_horizontales(etiquetas, distancias, "distancia al centroide", colores, ".3f")


def figura_distribucion(etiquetas: Sequence[str], quejas: Sequence[int]) -> go.Figure:
    """Cantidad de quejas del corpus en cada grupo."""
    return _barras_horizontales(etiquetas, quejas, "quejas del corpus", [COLOR_BARRA] * len(quejas), ",.0f")
