"""Pestaña "Cómo el agente crea las categorías": el mapa 2D de las quejas formando sus grupos, paso a paso.

Todo lo que se ve lo calcula el agente de verdad (:mod:`codigo.creacion`); solo el ritmo de
reproducción se puede frenar, y la pantalla lo dice. La primera vez se calcula en vivo y la
grabación queda en memoria: las demás veces se repite sin recalcular. El bucle de dibujo corre
dentro de un ``st.fragment`` y escribe en marcos ``st.empty``, así que solo se redibuja esta pestaña.
"""

from __future__ import annotations

import threading

import streamlit as st

from app import escena_creacion
from codigo import config, creacion
from codigo.decision_k import DecisionK
from codigo.triaje import AgenteTriaje

CLAVE_VELOCIDAD = "velocidad_creacion"
TEXTO_BOTON = "Ver al agente crear las categorías"


@st.cache_resource(show_spinner=False)
def _repositorio() -> dict:
    """Grabación compartida entre sesiones (el cálculo es determinista: semilla fija)."""
    return {"lock": threading.Lock(), "eventos": None}


@st.cache_resource(show_spinner="Preparando el mapa 2D de las quejas...")
def _mapa(_base: AgenteTriaje, _decision: DecisionK) -> creacion.Mapa:
    return creacion.preparar_mapa(_base, _decision)


def aviso_velocidad(velocidad: str) -> str:
    """Texto fijo que dice qué se está frenando (nunca los cálculos, solo el ritmo)."""
    paso, detalle = config.VELOCIDADES[velocidad]
    if paso is None:
        return ("**Velocidad real.** Cada paso se muestra tras esperar lo que tardó su cálculo de verdad; "
                "las iteraciones, que duran milisegundos, pasan casi sin pausa.")
    return (f"**Reproducción ralentizada ({velocidad}).** Los cálculos son reales y no cambian: solo se espacian los "
            f"pasos ({paso:g} s por paso, {detalle:g} s por detalle) para poder seguirlos.")


def avisos_fijos(mapa: creacion.Mapa) -> None:
    p = mapa.proyeccion
    if mapa.es_submuestra:
        st.caption(
            f"**Dibujo de una muestra fija de {len(mapa.indices):,} de las {mapa.n:,} quejas** (aleatoria, semilla "
            f"{config.SEMILLA}, la misma en todos los cuadros) para que la animación sea fluida. "
            "**Los cálculos usan las quejas completas**: inercia, cambios de grupo y tamaños son los reales.")
    st.caption(
        f"**El dibujo 2D explica solo ≈{p.varianza_explicada:.1%} de la varianza.** Los grupos se forman en "
        f"{config.N_COMPONENTES_SVD} dimensiones; al aplastarlos a 2D se ven mezclados.  \n"
        f"**Silueta 2D = {p.silueta_2d:.3f}: es la separación del dibujo 2D, no la calidad del modelo.** "
        f"La silueta del modelo en {config.N_COMPONENTES_SVD} dimensiones es {mapa.silueta_modelo:.4f}.  \n"
        f"**El modelo final es la mejor de {config.N_INIT} inicializaciones** (la de menor inercia); "
        "la animación muestra a esa ganadora.")


def _crear_marcos() -> dict:
    """Distribución de la pantalla: narración, fase 1 (mosaico + silueta) y fase 2 (mapa grande + paneles)."""
    marcos: dict = {"narracion": st.empty(), "mosaico": {}}
    st.markdown("##### Fase 1 · Elegir cuántas categorías (K)")
    izq, der = st.columns([2, 3])
    with izq:
        celdas = [c for _ in range(config.FILAS_MOSAICO) for c in st.columns(config.MINIATURAS_POR_FILA)]
        for k, celda in zip(config.RANGO_K, celdas):
            marcos["mosaico"][k] = celda.empty()
    with der:
        marcos["silueta"], marcos["motivos"] = st.empty(), st.empty()
    marcos["decision"] = st.empty()
    st.markdown("##### Fase 2 · Formar los grupos sobre el mapa 2D de las quejas")
    mapa, lateral = st.columns([3, 1.4])
    marcos["mapa"] = mapa.empty()
    with lateral:
        marcos["contadores"] = st.empty()
        st.markdown(f"**Las {config.N_INIT} inicializaciones**")
        marcos["tira"] = st.empty()
    st.markdown("##### Fase 3 · Ponerles nombre")
    marcos["nombres"] = st.empty()
    return marcos


def _grabar(generador, destino: list):
    """Pasa los eventos tal cual y los guarda; así la primera ejecución en vivo queda grabada."""
    for ev in generador:
        destino.append(ev)
        yield ev


def _nota_de_calculo(marco, grabado: list | None) -> None:
    if grabado is None:
        marco.caption("La primera vez el agente calcula todo en vivo (unos segundos); después se repite sin recalcular.")
    else:
        total = sum(creacion.duracion_por_fase(grabado).values())
        marco.caption(f"Ya calculado (**{total:.1f} s** de cálculo real): se repite sin recalcular.")


@st.fragment
def _escena(base: AgenteTriaje, decision: DecisionK) -> None:
    repo, mapa = _repositorio(), _mapa(base, decision)
    velocidades = list(config.VELOCIDADES)
    velocidad = st.radio("Velocidad de reproducción", velocidades, horizontal=True, key=CLAVE_VELOCIDAD,
                         index=velocidades.index(config.VELOCIDAD_INICIAL))
    lanzar = st.button(TEXTO_BOTON, type="primary")
    st.caption(aviso_velocidad(velocidad))
    grabado = repo["eventos"]
    nota = st.empty()
    _nota_de_calculo(nota, grabado)

    escena = escena_creacion.Escena(_crear_marcos(), mapa)
    avisos_fijos(mapa)
    if not lanzar:
        if grabado is None:
            escena.mostrar(creacion.Evento(0, "mapa", {"mapa": mapa, "n": mapa.n}))
            escena.m["narracion"].markdown("#### Pulsa el botón para ver cómo el agente crea las categorías")
        else:
            for ev in grabado:
                escena.aplicar(ev)
                escena.narrado = creacion.narracion(ev)
            escena.pintar_todo()
        return
    if grabado is None:
        with repo["lock"]:
            if repo["eventos"] is None:
                nuevos: list = []
                escena_creacion.reproducir(escena, _grabar(creacion.generar_eventos(base, decision, mapa), nuevos),
                                           velocidad, en_vivo=True)
                repo["eventos"] = nuevos
                _nota_de_calculo(nota, nuevos)
                return
            grabado = repo["eventos"]
    escena_creacion.reproducir(escena, grabado, velocidad, en_vivo=False)


def pestana_creacion(base: AgenteTriaje, decision: DecisionK) -> None:
    st.subheader("Cómo el agente crea las categorías")
    st.write(
        "Cada punto es una queja del corpus, dibujada en 2D. Verás al agente elegir cuántas categorías crear, "
        "formar los grupos paso a paso y ponerles nombre. **Todo se calcula de verdad**; solo el ritmo de la "
        "reproducción se puede frenar.")
    _escena(base, decision)
