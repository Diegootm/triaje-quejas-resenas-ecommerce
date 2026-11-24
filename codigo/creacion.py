"""Cómo el agente crea las categorías, como una secuencia de eventos REALES (sin Streamlit).

:func:`generar_eventos` hace el trabajo de verdad (barrido de K, 10 arranques de Lloyd, nombres)
y entrega un :class:`Evento` por cada paso visible. No se inventa ningún dato: cada evento
lleva lo que se calculó y ``duracion``, los segundos que tardó ese cálculo. La interfaz
solo decide con qué ritmo mostrarlos (:func:`espera`): nunca cambia lo calculado.

Fases: 0 = preparar el mapa 2D, 1 = elegir K, 2 = formar los grupos, 3 = ponerles nombre.
"""

from __future__ import annotations

import html
import textwrap
import time
from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np
import pandas as pd

from codigo import agrupamiento, config, lloyd
from codigo.decision_k import DecisionK, decidir_k, explicar_decision
from codigo.presentacion import etiqueta_grupo
from codigo.triaje import AgenteTriaje

TOLERANCIA_COMPROBACION = 1e-6


@dataclass(frozen=True)
class Evento:
    fase: int
    tipo: str
    datos: dict
    duracion: float = 0.0     # segundos reales que tardó el cálculo de este paso
    detalle: bool = False     # sub-paso fino: usa la pausa corta de la velocidad elegida


def espera(evento: Evento, velocidad: str, en_vivo: bool) -> float:
    """Segundos que se esperan ANTES de mostrar ``evento``.

    Ritmo real: al repetir se espera lo que tardó el cálculo; en vivo ya se esperó calculándolo.
    Velocidad lenta: una pausa fija; en vivo se descuenta lo que ya tardó el cálculo.
    """
    pausa_paso, pausa_detalle = config.VELOCIDADES[velocidad]
    pausa = pausa_detalle if evento.detalle else pausa_paso
    if pausa is None:
        return 0.0 if en_vivo or evento.detalle else evento.duracion
    return max(0.0, pausa - (evento.duracion if en_vivo else 0.0))


def fragmentos_resenas(textos: Sequence[str], largo: int = config.LARGO_FRAGMENTO_RESENA) -> list[str]:
    """Fragmento corto de cada reseña, escapado y con saltos ``<br>`` para el cursor del mapa."""
    salida = []
    for t in textos:
        t = " ".join(str(t).split())
        corto = t if len(t) <= largo else t[:largo].rsplit(" ", 1)[0] + "…"
        salida.append("<br>".join(textwrap.wrap(html.escape(corto), config.ANCHO_LINEA_HOVER)))
    return salida


@dataclass(frozen=True)
class Mapa:
    """El dibujo 2D fijo sobre el que se pinta todo (se calcula una vez)."""

    proyeccion: lloyd.Proyeccion2D
    fragmentos: list[str]         # fragmento de reseña de cada punto, para el cursor
    k: int
    n: int
    silueta_modelo: float         # silueta del modelo en las 100 dimensiones (de seleccion_k.json)
    duracion: float
    indices: np.ndarray           # quejas que se DIBUJAN (muestra fija); los cálculos usan todas

    @property
    def puntos2d(self) -> np.ndarray:
        return self.proyeccion.puntos2d[self.indices]

    @property
    def fragmentos_dibujo(self) -> list[str]:
        return [self.fragmentos[i] for i in self.indices]

    @property
    def es_submuestra(self) -> bool:
        return len(self.indices) < self.n


def preparar_mapa(base: AgenteTriaje, decision: DecisionK) -> Mapa:
    """Plano 2D fijo del dibujo: el que mejor separa los grupos del modelo guardado."""
    t0 = time.perf_counter()
    X, centros = base.X_svd_corpus, [base.kmeans_referencia.cluster_centers_]
    proy = lloyd.elegir_proyeccion([
        lloyd.proyectar_pca_puntos(X, centros, base.etiquetas_referencia),
        lloyd.proyectar_pca_centroides(X, centros, base.etiquetas_referencia),
    ])
    fragmentos = fragmentos_resenas(base.corpus["Review Text"].fillna(""))
    n = len(X)
    indices = np.sort(np.random.RandomState(config.SEMILLA).choice(n, min(config.SUBMUESTRA_MAPA, n), replace=False))
    return Mapa(proy, fragmentos, base.k_referencia, n, decision.silueta_de(base.k_referencia),
                time.perf_counter() - t0, indices)


def comparar_con_guardado(tabla_viva: pd.DataFrame, decision: DecisionK) -> dict:
    """Compara el barrido recién calculado con ``seleccion_k.json``."""
    viva = tabla_viva.set_index("K").sort_index()
    guardada = decision.tabla.set_index("K").sort_index()
    mismos_k = list(viva.index) == list(guardada.index)
    iguales = mismos_k and all(
        np.allclose(viva[c], guardada[c], rtol=TOLERANCIA_COMPROBACION, atol=TOLERANCIA_COMPROBACION)
        for c in ("inercia", "silueta"))
    k_regla = decidir_k(tabla_viva).k_elegido
    return {"tabla_igual": bool(iguales), "k_regla": k_regla, "k_guardado": decision.k_elegido,
            "coincide": bool(iguales and k_regla == decision.k_elegido)}


def generar_eventos(base: AgenteTriaje, decision: DecisionK, mapa: Mapa | None = None) -> Iterator[Evento]:
    """Calcula de verdad y entrega los eventos en el orden en que se muestran."""
    X, k, modelo = base.X_svd_corpus, base.k_referencia, base.kmeans_referencia
    n = len(X)

    # --- Fase 0: el dibujo 2D (se calcula una vez; todo se pinta sobre él) ------------------
    mapa = mapa or preparar_mapa(base, decision)
    proy = mapa.proyeccion
    yield Evento(0, "mapa", {"mapa": mapa, "n": n}, mapa.duracion)

    # --- Fase 1: elegir K -------------------------------------------------------------------
    filas, barrido = [], agrupamiento.iterar_barrido_k(X, list(config.RANGO_K))
    for kk in config.RANGO_K:
        yield Evento(1, "k_probando", {"k": kk}, detalle=True)
        t0 = time.perf_counter()
        fila, km = next(barrido)
        filas.append(fila)
        motivo = agrupamiento.recomendar_k(pd.DataFrame([fila])).descartados.get(kk)
        yield Evento(1, "k_listo", {**fila, "etiquetas": km.labels_.astype(np.int8), "motivo": motivo},
                     time.perf_counter() - t0)
    tabla = pd.DataFrame(filas)
    t0 = time.perf_counter()
    viva = decidir_k(tabla)
    yield Evento(1, "k_decidido", {
        "k": viva.k_elegido, "umbral": viva.umbral, "candidatos": viva.candidatos, "descartados": viva.descartados,
        "silueta_mejor": viva.silueta_mejor, "explicacion": explicar_decision(viva),
        "comprobacion": comparar_con_guardado(tabla, decision),
    }, time.perf_counter() - t0)

    # --- Fase 2: formar los grupos (10 inicializaciones de Lloyd; se anima la ganadora) -----
    yield Evento(2, "grises", {"n": n})
    arranques, iterador = [], lloyd.iterar_arranques(X, k)
    for i in range(config.N_INIT):
        yield Evento(2, "arranque_probando", {"indice": i, "total": config.N_INIT}, detalle=True)
        t0 = time.perf_counter()
        a = next(iterador)
        arranques.append(a)
        ari = lloyd.ari_trayectoria_vs_modelo(a.final.etiquetas, modelo)
        yield Evento(2, "arranque_listo", {"indice": i, "inercia": a.inercia, "iteraciones": a.iteraciones,
                                           "ari": ari}, time.perf_counter() - t0)
    ganador = lloyd.elegir_ganador(arranques)
    yield Evento(2, "ganador", {"indice": ganador.indice, "inercia": ganador.inercia,
                                "inercia_modelo": float(modelo.inertia_)})
    inicial = ganador.trayectoria[0]
    for j, centro in enumerate(inicial.centros):          # k-means++ los devuelve en el orden en que los eligió
        yield Evento(2, "semilla", {"j": j, "total": k, "centro2d": proy.proyectar(centro)})
    for est in ganador.trayectoria:
        yield Evento(2, "iteracion", {
            "iteracion": est.iteracion, "etiquetas": est.etiquetas.astype(np.int8), "inercia": est.inercia,
            "cambios": est.cambios, "centros2d": proy.proyectar(est.centros)})
    yield Evento(2, "estable", {"iteraciones": ganador.iteraciones, "inercia": ganador.inercia,
                                "ari": lloyd.ari_trayectoria_vs_modelo(ganador.final.etiquetas, modelo)})

    # --- Fase 3: ponerles nombre ------------------------------------------------------------
    tamanos = np.bincount(base.etiquetas_referencia, minlength=k)
    centros2d = proy.proyectar(ganador.final.centros)
    for g in range(k):
        yield Evento(3, "nombre_inicio", {"g": g, "centro2d": centros2d[g], "total": int(tamanos[g])})
        for i, termino in enumerate(base.terminos[g]):
            yield Evento(3, "termino", {"g": g, "i": i, "termino": termino}, detalle=True)
        for paso in range(1, config.PASOS_CONTEO + 1):
            yield Evento(3, "conteo", {"g": g, "n": round(int(tamanos[g]) * paso / config.PASOS_CONTEO)}, detalle=True)
        yield Evento(3, "nombre_listo", {"g": g, "nombre": base.nombres[g],
                                         "nombre_informe": base.nombres_secundarios.get(g)})
    yield Evento(3, "fin", {"k": k, "n": n})


def narracion(ev: Evento) -> str:
    """La línea que dice qué está haciendo el agente en este paso."""
    d = ev.datos
    por_tipo = {
        "mapa": lambda: f"Preparando el mapa 2D de {d['n']:,} quejas…",
        "k_probando": lambda: f"Probando {d['k']} categorías…",
        "k_listo": lambda: f"{d['K']} categorías probadas: silueta {d['silueta']:.4f}",
        "k_decidido": lambda: f"Decidido: {d['k']} categorías",
        "grises": lambda: f"{d['n']:,} quejas todavía sin categoría",
        "arranque_probando": lambda: f"Probando la inicialización {d['indice'] + 1} de {d['total']}…",
        "arranque_listo": lambda: f"Inicialización {d['indice'] + 1}: inercia final {d['inercia']:.2f}",
        "ganador": lambda: f"La inicialización {d['indice'] + 1} quedó con la menor inercia: es la que se anima",
        "semilla": lambda: f"Colocando el centroide {d['j'] + 1} de {d['total']} (k-means++)…",
        "iteracion": lambda: ("Asignando cada queja al centroide más cercano…" if d["iteracion"] == 0 else
                              f"Iteración {d['iteracion']}: reasignando {d['cambios']:,} quejas…"),
        "estable": lambda: f"Las categorías se estabilizaron tras {d['iteraciones']} iteraciones",
        "nombre_inicio": lambda: f"Nombrando el {etiqueta_grupo(d['g'])}…",
        "termino": lambda: f"{etiqueta_grupo(d['g'])}: término «{d['termino']}»",
        "conteo": lambda: f"{etiqueta_grupo(d['g'])}: contando sus quejas…",
        "nombre_listo": lambda: f"{etiqueta_grupo(d['g'])} listo",
        "fin": lambda: f"Listo: {d['k']} categorías creadas y nombradas",
    }
    return por_tipo[ev.tipo]()


def duracion_por_fase(eventos: Sequence[Evento]) -> dict[int, float]:
    """Segundos reales de cálculo por fase (suma de ``duracion``)."""
    total: dict[int, float] = {}
    for ev in eventos:
        total[ev.fase] = total.get(ev.fase, 0.0) + ev.duracion
    return total
