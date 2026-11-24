"""Estado de la app: modelos por K, locks y K activo (el del agente o uno manual).

``Almacen`` es Python puro (sin Streamlit) para poder probarlo. Locks:

- ``_lock_dict``: muy corto; solo protege las escrituras en los diccionarios compartidos.
- un ``RLock`` por K: entrenar un K no bloquea a las sesiones que usan otro K.
- ``_lock_barrido``: un solo barrido a la vez; no bloquea el entrenamiento de un K ni el triaje.
"""

from __future__ import annotations

import threading
from typing import Callable

import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans

from codigo import agrupamiento, config
from codigo.decision_k import ORIGEN_MANUAL, DecisionK, cargar_decision
from codigo.preprocesamiento import COMANDO_PREPARAR_ENTORNO, recursos_nltk_faltantes
from codigo.triaje import AgenteTriaje

CLAVE_K = "k_activo"
COMANDO_ENTRENAR = "python entrenar_agente.py"


class Almacen:
    """Memoria compartida entre sesiones: modelos K-means y agentes por K, y la tabla del barrido."""

    def __init__(self) -> None:
        self._lock_dict = threading.Lock()
        self._lock_barrido = threading.Lock()
        self._locks_k: dict[int, threading.RLock] = {}
        self.modelos: dict[int, KMeans] = {}
        self.agentes: dict[int, AgenteTriaje] = {}
        self.barrido: pd.DataFrame | None = None

    def _lock_de(self, k: int) -> threading.RLock:
        with self._lock_dict:
            return self._locks_k.setdefault(k, threading.RLock())

    def agente_listo(self, base: AgenteTriaje, k: int) -> bool:
        return k == base.k_referencia or k in self.agentes

    def modelo_para_k(self, base: AgenteTriaje, k: int) -> KMeans:
        """Modelo guardado si K coincide; si no, el de memoria o uno nuevo (entrenado bajo el lock de ese K)."""
        if k == base.k_referencia:
            return base.kmeans_referencia
        with self._lock_de(k):
            if k not in self.modelos:
                modelo = agrupamiento.entrenar_kmeans(base.X_svd_corpus, k=k)
                with self._lock_dict:
                    self.modelos[k] = modelo
            return self.modelos[k]

    def agente_para_k(self, base: AgenteTriaje, k: int) -> AgenteTriaje:
        if k == base.k_referencia:
            return base
        with self._lock_de(k):
            if k not in self.agentes:
                agente = base.con_kmeans(self.modelo_para_k(base, k))
                with self._lock_dict:
                    self.agentes[k] = agente
            return self.agentes[k]

    def calcular_barrido(
        self, base: AgenteTriaje, progreso: Callable[[int, int, dict], None] | None = None,
    ) -> pd.DataFrame:
        """Barrido K = 2..12 (una sola vez). Los modelos entrenados quedan disponibles para elegir cualquier K."""
        with self._lock_barrido:
            if self.barrido is None:
                tabla, modelos = agrupamiento.barrido_k(base.X_svd_corpus, list(config.RANGO_K), progreso=progreso)
                with self._lock_dict:
                    for k, modelo in modelos.items():
                        self.modelos.setdefault(k, modelo)
                    self.barrido = tabla
            return self.barrido


@st.cache_resource(show_spinner=False)
def almacen() -> Almacen:
    return Almacen()


@st.cache_resource(show_spinner="Cargando modelos entrenados...")
def cargar_agente() -> AgenteTriaje:
    return AgenteTriaje()


def preparar_o_detener() -> tuple[AgenteTriaje, DecisionK]:
    """Comprueba NLTK, la decisión de K y los modelos; si falta algo, muestra un error claro y detiene la app."""
    faltantes = recursos_nltk_faltantes()
    if faltantes:
        st.error(f"Faltan recursos de NLTK: {', '.join(faltantes)}. Instálalos una vez con:")
        st.code(COMANDO_PREPARAR_ENTORNO, language="bash")
        st.stop()
    try:
        decision = cargar_decision(config.DIR_MODELOS)
        return cargar_agente(), decision
    except FileNotFoundError as error:
        st.error(str(error))
        st.code(COMANDO_ENTRENAR, language="bash")
        st.stop()


# --------------------------------------------------------------------------
# K activo: el del agente o uno manual
# --------------------------------------------------------------------------
def usar_k(k: int) -> None:
    """Callback de botones: fija el K activo."""
    st.session_state[CLAVE_K] = k


def selector_k(base: AgenteTriaje, decision: DecisionK) -> int:
    """Barra lateral: K elegido por el agente frente a K manual. Devuelve el K activo (válido)."""
    k_agente = decision.k_elegido
    if CLAVE_K not in st.session_state:
        st.session_state[CLAVE_K] = k_agente
    with st.sidebar:
        st.header("Número de grupos")
        origen = " (impuesto con `--k`)" if decision.origen == ORIGEN_MANUAL else ""
        st.markdown(f"**K elegido por el agente: {k_agente}**{origen}")
        k = st.slider(
            "K manual (exploración)", config.K_MIN, config.K_MAX, key=CLAVE_K,
            help=("Mover el control solo explora otros K: entrena K-means sobre el mismo TF-IDF + SVD y las "
                  "quejas nuevas se asignan con ese K. El agente sigue eligiendo K con su regla."),
        )
        if k == k_agente:
            st.caption("Estás usando el K del agente.")
        else:
            st.markdown(f"**K manual: {k}**")
            st.button(f"Volver al K del agente ({k_agente})", on_click=usar_k, args=(k_agente,))
    return agrupamiento.validar_k(k, len(base.corpus))


def agente_activo(base: AgenteTriaje, k: int) -> AgenteTriaje:
    """Agente con el K activo; si hay que entrenar ese K, lo muestra con un spinner."""
    memoria = almacen()
    if memoria.agente_listo(base, k):
        return memoria.agente_para_k(base, k)
    with st.spinner(f"Entrenando K-means con K = {k} sobre {len(base.corpus):,} quejas..."):
        return memoria.agente_para_k(base, k)
