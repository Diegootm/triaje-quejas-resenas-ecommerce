"""Interfaz Streamlit del agente de triaje de quejas.

Carga los modelos entrenados (vectorizador TF-IDF, SVD y K-means) y, para cada queja
nueva, sugiere el grupo más cercano junto con sus términos representativos y las reseñas
más parecidas del corpus.

El número de grupos K lo elige el propio agente al entrenar (modelos/seleccion_k.json). La
barra lateral permite explorar otro K de forma manual: solo se entrena K-means sobre la
representación TF-IDF + SVD ya calculada. Clasificar una queja nueva nunca reentrena.

Ejecución local (desde la raíz del repositorio):

    streamlit run app/app_streamlit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from app import estado, vista_creacion, vista_k, vista_triaje  # noqa: E402


def _configurar_pagina() -> None:
    st.set_page_config(page_title="Triaje de quejas de reseñas", page_icon=":material/space_dashboard:", layout="wide")


def main() -> None:
    _configurar_pagina()
    st.title("Agente de triaje de quejas en reseñas de productos")
    st.write("Aprendizaje no supervisado (TF-IDF + SVD + K-means) sobre *Women's E-Commerce Clothing Reviews*.")
    base, decision = estado.preparar_o_detener()

    k = estado.selector_k(base, decision)
    agente = estado.agente_activo(base, k)
    vista_triaje.barra_lateral(agente, decision)
    t1, t2, t3, t4, t5 = st.tabs(
        ["Queja individual", "Lote CSV", "Grupos descubiertos", "Selección de K", "Cómo el agente crea las categorías"])
    with t1:
        vista_triaje.pestana_texto(agente)
    with t2:
        vista_triaje.pestana_lote(agente)
    with t3:
        vista_triaje.pestana_grupos(agente)
    with t4:
        vista_k.pestana_k(base, agente, decision)
    with t5:
        vista_creacion.pestana_creacion(base, decision)


if __name__ == "__main__":
    main()
