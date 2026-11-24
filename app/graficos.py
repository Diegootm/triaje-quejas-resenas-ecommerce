"""Figuras de Plotly como funciones puras (sin Streamlit): reciben datos y devuelven ``go.Figure``.

Los gráficos de barras, las miniaturas y la curva de silueta son ESTÁTICOS: sin zoom,
desplazamiento, arrastre ni barra de herramientas (``config=CONFIG_ESTATICO``). El mapa
grande solo es interactivo al final, para leer un fragmento de reseña al pasar el cursor
(``config=CONFIG_MAPA``); mientras se anima no lleva datos de cursor.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import plotly.graph_objects as go

from codigo import config

CONFIG_ESTATICO = {"staticPlot": True, "displayModeBar": False}
CONFIG_MAPA = {"displayModeBar": False, "scrollZoom": False}   # sin zoom ni arrastre; solo el cursor

COLOR_BARRA = "#9DB4CE"
COLOR_DESTACADO = "#E45756"
MARGEN_EJE = 1.1            # el eje X llega hasta el máximo por este factor (y siempre empieza en 0)
ALTO_MINIMO = 240
ALTO_POR_BARRA = 28
ALTO_POR_BARRA_COMPACTA = 22

COLOR_SIN_GRUPO = "#C8CCD2"
COLOR_CANDIDATO = "#54A24B"
COLOR_DESCARTADO = "#B0B4BA"
PALETA_GRUPOS = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2", "#9D755D",
                 "#72B7B2", "#EECA3B", "#FF9DA6", "#2F4B7C", "#A0CBE8", "#8CD17D"]


def _barras_horizontales(
    etiquetas: Sequence[str], valores: Sequence[float], titulo_x: str,
    colores: Sequence[str], formato_texto: str,
    textos: Sequence[str] | None = None, alto_por_barra: int = ALTO_POR_BARRA, margen_derecho: int = 40,
) -> go.Figure:
    """Barras horizontales estáticas; el eje X va de 0 a ``max · MARGEN_EJE`` (sin ticks negativos)."""
    maximo = max((float(v) for v in valores), default=0.0)
    fig = go.Figure(go.Bar(
        x=list(valores), y=list(etiquetas), orientation="h", marker_color=list(colores),
        text=list(textos) if textos is not None else [format(float(v), formato_texto) for v in valores],
        textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x}<extra></extra>",
    ))
    fig.update_xaxes(range=[0, maximo * MARGEN_EJE if maximo > 0 else 1.0], fixedrange=True, title=titulo_x,
                     rangemode="nonnegative", showgrid=True)
    fig.update_yaxes(autorange="reversed", fixedrange=True, automargin=True)   # el primer grupo arriba
    fig.update_layout(
        dragmode=False, showlegend=False, bargap=0.25,
        height=max(ALTO_MINIMO if textos is None else 0, alto_por_barra * len(etiquetas) + 80),
        margin=dict(l=10, r=margen_derecho, t=10, b=40),
    )
    return fig


def figura_distancias(etiquetas: Sequence[str], distancias: Sequence[float], indice_sugerido: int) -> go.Figure:
    """Distancia de una queja a cada centroide (menor = más parecido); el grupo sugerido va destacado."""
    colores = [COLOR_DESTACADO if i == indice_sugerido else COLOR_BARRA for i in range(len(distancias))]
    return _barras_horizontales(etiquetas, distancias, "distancia al centroide", colores, ".3f")


def figura_distribucion(etiquetas: Sequence[str], quejas: Sequence[int]) -> go.Figure:
    """Cantidad de quejas del corpus en cada grupo."""
    return _barras_horizontales(etiquetas, quejas, "quejas del corpus", [COLOR_BARRA] * len(quejas), ",.0f")


def figura_inicializaciones(corridas: Sequence[Mapping], indice_ganador: int | None = None) -> go.Figure:
    """Inercia final de cada inicialización de K-means (menor = mejor) y su ARI frente al modelo guardado.

    El eje empieza en 0 a propósito: las inicializaciones difieren muy poco entre sí y no se exagera.
    """
    etiquetas = [f"Inicialización {c['indice'] + 1}" for c in corridas]
    colores = [COLOR_DESTACADO if c["indice"] == indice_ganador else COLOR_BARRA for c in corridas]
    textos = [f"{c['inercia']:.2f} · ARI {c['ari']:.2f}" for c in corridas]
    return _barras_horizontales(etiquetas, [c["inercia"] for c in corridas], "inercia final", colores, ".2f",
                                textos=textos, alto_por_barra=ALTO_POR_BARRA_COMPACTA, margen_derecho=110)


# --------------------------------------------------------------------------
# Mapa 2D de las quejas
# --------------------------------------------------------------------------
def _escala_discreta(k: int) -> list[list]:
    """Escala de color con un tramo plano por grupo, para colorear con el entero del grupo."""
    return [[p, PALETA_GRUPOS[g % len(PALETA_GRUPOS)]] for g in range(k) for p in (g / k, (g + 1) / k)]


def _puntos(puntos: np.ndarray, etiquetas: Sequence[int] | None, k: int, tamano: int, hover: Sequence[str] | None,
            webgl: bool):
    x, y = np.round(puntos[:, 0], 3), np.round(puntos[:, 1], 3)
    marcador = dict(size=tamano, opacity=0.8, line=dict(width=0))
    if etiquetas is None:
        marcador["color"] = COLOR_SIN_GRUPO
    else:
        marcador.update(color=np.asarray(etiquetas, dtype=int), colorscale=_escala_discreta(k),
                        cmin=-0.5, cmax=k - 0.5, showscale=False)
    extra = dict(hoverinfo="skip") if hover is None else dict(hovertext=list(hover), hoverinfo="text")
    traza = go.Scattergl if webgl else go.Scatter
    return traza(x=x, y=y, mode="markers", marker=marcador, showlegend=False, **extra)


def _ejes_fijos(fig: go.Figure, puntos: np.ndarray) -> None:
    """Rango fijo (el mismo en cada cuadro) y sin ejes: solo importa la posición relativa."""
    for eje, columna in ((fig.update_xaxes, 0), (fig.update_yaxes, 1)):
        bajo, alto = float(puntos[:, columna].min()), float(puntos[:, columna].max())
        pad = 0.04 * (alto - bajo)
        eje(range=[bajo - pad, alto + pad], visible=False, fixedrange=True)


def figura_miniatura(puntos: np.ndarray, etiquetas: Sequence[int], k: int, destacada: bool = False) -> go.Figure:
    """Mapa 2D diminuto coloreado por la partición con ``k`` grupos (sin ejes ni interacción)."""
    fig = go.Figure(_puntos(puntos, etiquetas, k, config.TAMANO_PUNTO_MINIATURA, None, webgl=False))
    _ejes_fijos(fig, puntos)
    fig.add_annotation(text=f"<b>K = {k}</b>", x=0.02, y=0.98, xref="paper", yref="paper", xanchor="left",
                       yanchor="top", showarrow=False, font=dict(size=11), bgcolor="rgba(255,255,255,0.8)")
    if destacada:
        fig.add_shape(type="rect", xref="paper", yref="paper", x0=0, y0=0, x1=1, y1=1,
                      line=dict(color=COLOR_DESTACADO, width=4))
    fig.update_layout(height=config.MINIATURA_ALTO_PX, margin=dict(l=0, r=0, t=0, b=0), dragmode=False,
                      plot_bgcolor="white")
    return fig


def figura_mapa(
    puntos: np.ndarray,
    etiquetas: Sequence[int] | None = None,
    k: int = 0,
    centros: np.ndarray | None = None,
    estela: Sequence[np.ndarray] = (),
    rotulos: Mapping[int, str] | None = None,
    fragmentos: Sequence[str] | None = None,
) -> go.Figure:
    """Mapa 2D grande. Sin ``etiquetas`` todos los puntos son grises (aún sin categoría).

    ``centros`` (m, 2): centroides actuales (el centroide ``g`` usa el color del grupo ``g``);
    ``estela``: posiciones anteriores, de más antigua a más reciente, que dejan una línea corta;
    ``rotulos``: ``{grupo: texto}`` colocado en su centroide; ``fragmentos``: texto del cursor
    por punto (solo se pasa cuando termina la animación, para no mandar ese peso en cada cuadro).
    """
    textos = None
    if fragmentos is not None:
        textos = list(fragmentos) if etiquetas is None else [
            f"<b>Grupo {int(g) + config.PRIMER_GRUPO_VISIBLE}</b><br>{f}" for g, f in zip(etiquetas, fragmentos)]
    fig = go.Figure(_puntos(puntos, etiquetas, k, config.TAMANO_PUNTO_MAPA, textos, webgl=config.MAPA_WEBGL))
    if centros is not None and len(centros):
        trayecto = [*estela, centros]
        for g in range(len(centros)):
            color = PALETA_GRUPOS[g % len(PALETA_GRUPOS)]
            linea = np.array([t[g] for t in trayecto if len(t) > g])
            if len(linea) > 1:
                fig.add_trace(go.Scatter(x=linea[:, 0], y=linea[:, 1], mode="lines", hoverinfo="skip",
                                         line=dict(color=color, width=3), showlegend=False))
            fig.add_trace(go.Scatter(
                x=[centros[g][0]], y=[centros[g][1]], mode="markers+text", text=[str(g + config.PRIMER_GRUPO_VISIBLE)],
                textposition="top center", hoverinfo="skip", showlegend=False, textfont=dict(size=13, color="#222"),
                marker=dict(size=16, color=color, symbol="diamond", line=dict(color="#222", width=2))))
    for g, texto in (rotulos or {}).items():
        x, y = centros[g]
        fig.add_annotation(x=x, y=y, text=texto, showarrow=True, arrowhead=0, ax=0, ay=-34, font=dict(size=12),
                           bgcolor="rgba(255,255,255,0.9)", bordercolor=PALETA_GRUPOS[g % len(PALETA_GRUPOS)],
                           borderwidth=2, borderpad=3)
    _ejes_fijos(fig, puntos)
    fig.update_layout(height=config.MAPA_ALTO_PX, margin=dict(l=0, r=0, t=0, b=0), dragmode=False,
                      plot_bgcolor="white", hovermode="closest" if fragmentos is not None else False)
    return fig


def figura_silueta(
    ks: Sequence[int], siluetas: Sequence[float], descartados: Mapping[int, str],
    umbral: float | None = None, k_elegido: int | None = None,
) -> go.Figure:
    """Silueta de cada K a medida que se entrena. Los K descartados van en gris (su motivo se escribe aparte).

    Cuando ya se decidió (``umbral``), los K que superan el umbral se resaltan y ``k_elegido`` va en rojo.
    """
    colores, tamanos, simbolos = [], [], []
    for k, s in zip(ks, siluetas):
        if k in descartados:
            estilo = (COLOR_DESCARTADO, 9, "x")
        elif k == k_elegido:
            estilo = (COLOR_DESTACADO, 16, "circle")
        elif umbral is not None and s >= umbral:
            estilo = (COLOR_CANDIDATO, 12, "circle")
        else:
            estilo = (COLOR_BARRA, 10, "circle")
        colores.append(estilo[0]), tamanos.append(estilo[1]), simbolos.append(estilo[2])
    fig = go.Figure(go.Scatter(
        x=list(ks), y=list(siluetas), mode="lines+markers+text", line=dict(color=COLOR_DESCARTADO, width=1),
        marker=dict(color=colores, size=tamanos, symbol=simbolos), text=[f"{s:.4f}" for s in siluetas],
        textposition="top center", textfont=dict(size=9), showlegend=False, hoverinfo="skip"))
    if umbral is not None:
        fig.add_hline(y=umbral, line=dict(color=COLOR_DESTACADO, dash="dash", width=1.5),
                      annotation_text=f"umbral {umbral:.4f}", annotation_position="bottom right")
    bajo, alto = min(0.0, min(siluetas, default=0.0)), max(siluetas, default=0.0)
    ancho = (alto - bajo) or 1.0
    fig.update_xaxes(range=[config.K_MIN - 0.5, config.K_MAX + 0.5], dtick=1, title="K (número de categorías)",
                     fixedrange=True)
    fig.update_yaxes(range=[bajo - 0.15 * ancho, alto + 0.35 * ancho], title="silueta", fixedrange=True)
    fig.update_layout(height=config.FILAS_MOSAICO * config.MINIATURA_ALTO_PX + 60, margin=dict(l=10, r=10, t=10, b=10), dragmode=False)
    return fig
