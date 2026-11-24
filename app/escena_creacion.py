"""La "escena" de la pestaña de creación: marcos de pantalla (``st.empty``) y lo que se pinta en cada evento.

``Escena.mostrar(evento)`` actualiza el estado acumulado y redibuja SOLO lo que cambia. Todo lo que
se dibuja viene de los eventos de :mod:`codigo.creacion` (cálculos reales); aquí no se calcula nada.
Las figuras son funciones puras de :mod:`app.graficos`.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Iterable

import numpy as np
import streamlit as st

from app import graficos
from codigo import config
from codigo.creacion import Evento, Mapa, espera, narracion
from codigo.presentacion import etiqueta_grupo


class Escena:
    """``marcos``: ``narracion, mosaico {K: marco}, silueta, motivos, decision, mapa, contadores, tira, nombres``."""

    def __init__(self, marcos: dict[str, Any], mapa: Mapa):
        self.m = marcos
        self.mapa = mapa
        self.etiquetas_k: dict[int, np.ndarray] = {}
        self.ks: list[int] = []
        self.siluetas: list[float] = []
        self.descartados: dict[int, str] = {}
        self.decidido: dict | None = None
        self.corridas: list[dict] = []
        self.ganador: int | None = None
        self.etiquetas: np.ndarray | None = None
        self.centros = np.empty((0, 2))
        self.estela: deque = deque(maxlen=config.ITERACIONES_ESTELA)
        self.contadores: tuple[int, float, int] | None = None
        self.rotulos: dict[int, str] = {}
        self.tarjetas: dict[int, dict] = {}
        self.narrado = ""
        self._cuadros = 0

    # ---- estado --------------------------------------------------------------------------
    def aplicar(self, ev: Evento) -> None:
        d = ev.datos
        if ev.tipo == "k_listo":
            self.ks.append(d["K"]), self.siluetas.append(d["silueta"])
            self.etiquetas_k[d["K"]] = d["etiquetas"]
            if d["motivo"]:
                self.descartados[d["K"]] = d["motivo"]
        elif ev.tipo == "k_decidido":
            self.decidido = d
        elif ev.tipo == "grises":
            self.etiquetas, self.centros = None, np.empty((0, 2))
            self.estela.clear()
        elif ev.tipo == "arranque_listo":
            self.corridas.append({k: d[k] for k in ("indice", "inercia", "iteraciones", "ari")})
        elif ev.tipo == "ganador":
            self.ganador = d["indice"]
        elif ev.tipo == "semilla":
            self.centros = np.vstack([self.centros, d["centro2d"]])
        elif ev.tipo == "iteracion":
            if self.etiquetas is not None:
                self.estela.append(self.centros)
            self.etiquetas, self.centros = d["etiquetas"], d["centros2d"]
            self.contadores = (d["iteracion"], d["inercia"], d["cambios"])
        elif ev.tipo == "nombre_inicio":
            self.tarjetas[d["g"]] = {"terminos": [], "n": 0, "total": d["total"], "listo": False}
            self.estela.clear()
            self.rotulos[d["g"]] = etiqueta_grupo(d["g"])
        elif ev.tipo == "termino":
            self.tarjetas[d["g"]]["terminos"].append(d["termino"])
        elif ev.tipo == "conteo":
            self.tarjetas[d["g"]]["n"] = d["n"]
        elif ev.tipo == "nombre_listo":
            self.tarjetas[d["g"]].update(listo=True, nombre=d["nombre"], informe=d["nombre_informe"])
            self.rotulos[d["g"]] = d["nombre"]

    # ---- dibujo --------------------------------------------------------------------------
    def _dibujar(self, marco, fig, config_plotly: dict) -> None:
        """Pinta ``fig`` en ``marco``; la clave única evita ids repetidos cuando dos cuadros son idénticos."""
        self._cuadros += 1
        marco.plotly_chart(fig, use_container_width=True, config=config_plotly, key=f"creacion-{self._cuadros}")

    def mostrar(self, ev: Evento) -> None:
        self.aplicar(ev)
        self.narrado = narracion(ev)
        self.m["narracion"].markdown(f"#### {self.narrado}")
        if ev.tipo == "k_listo":
            self._miniatura(ev.datos["K"])
            self._silueta()
        elif ev.tipo == "k_decidido":
            self._miniatura(ev.datos["k"], destacada=True)
            self._silueta()
            self._decision()
        elif ev.tipo in ("mapa", "grises", "semilla", "iteracion"):
            self._mapa()
        elif ev.tipo in ("arranque_listo", "ganador"):
            self._tira()
        elif ev.tipo in ("nombre_inicio", "nombre_listo"):
            self._mapa()
        elif ev.tipo == "fin":
            self._mapa(final=True)
        if ev.tipo == "iteracion" or ev.tipo == "semilla":
            self._contadores()
        if ev.tipo in ("nombre_inicio", "termino", "conteo", "nombre_listo"):
            self._tarjetas()

    def pintar_todo(self) -> None:
        """Estado final ya calculado (sin animar): todo lo que dejó la grabación."""
        for k in self.ks:
            self._miniatura(k, destacada=self.decidido is not None and k == self.decidido["k"])
        self._silueta()
        self._decision()
        if self.corridas:
            self._tira()
        self._contadores()
        self._tarjetas()
        self._mapa(final=True)
        self.m["narracion"].markdown(f"#### {self.narrado}")

    def _tira(self) -> None:
        self._dibujar(self.m["tira"], graficos.figura_inicializaciones(self.corridas, self.ganador),
                      graficos.CONFIG_ESTATICO)

    def _miniatura(self, k: int, destacada: bool = False) -> None:
        fig = graficos.figura_miniatura(self.mapa.puntos2d, self.etiquetas_k[k][self.mapa.indices], k, destacada)
        self._dibujar(self.m["mosaico"][k], fig, graficos.CONFIG_ESTATICO)

    def _silueta(self) -> None:
        d = self.decidido
        fig = graficos.figura_silueta(self.ks, self.siluetas, self.descartados,
                                      d["umbral"] if d else None, d["k"] if d else None)
        self._dibujar(self.m["silueta"], fig, graficos.CONFIG_ESTATICO)
        self.m["motivos"].caption("  \n".join(f"**K = {k}** descartado: {m}" for k, m in sorted(self.descartados.items())))

    def _decision(self) -> None:
        d = self.decidido
        if d is None:
            return
        c = d["comprobacion"]
        if c["coincide"]:
            veredicto = (f"✔ Coincide con modelos/{config.ARCHIVO_DECISION_K}: el barrido recién calculado repite la tabla "
                         f"guardada (inercia y silueta) y la regla vuelve a dar K = {c['k_guardado']}.")
        else:
            veredicto = (f"⚠ No coincide con modelos/{config.ARCHIVO_DECISION_K}: tabla igual = {c['tabla_igual']}; "
                         f"la regla da K = {c['k_regla']} y el archivo guarda K = {c['k_guardado']}.")
        with self.m["decision"].container():
            st.info(d["explicacion"])
            st.caption(veredicto)

    def _mapa(self, final: bool = False) -> None:
        fig = graficos.figura_mapa(
            self.mapa.puntos2d, None if self.etiquetas is None else self.etiquetas[self.mapa.indices], self.mapa.k,
            self.centros, list(self.estela), self.rotulos, self.mapa.fragmentos_dibujo if final else None)
        self._dibujar(self.m["mapa"], fig, graficos.CONFIG_MAPA if final else graficos.CONFIG_ESTATICO)

    def _contadores(self) -> None:
        if self.contadores is None:
            return
        it, inercia, cambios = self.contadores
        self.m["contadores"].markdown(
            f"**Iteración** {it}  \n**Inercia** {inercia:,.2f}  \n**Quejas que cambiaron de grupo** {cambios:,}")

    def _tarjetas(self) -> None:
        lineas = []
        for g, t in sorted(self.tarjetas.items()):
            cabecera = f"**{etiqueta_grupo(g)}** · {t['n']:,} de {t['total']:,} quejas"
            terminos = ", ".join(t["terminos"]) or "…"
            if t["listo"]:
                informe = f"  \n_Nombre del informe:_ {t['informe']}" if t["informe"] else ""
                lineas.append(f"{cabecera}  \n_Nombre automático:_ {t['nombre']}{informe}  \n_Términos:_ {terminos}")
            else:
                lineas.append(f"{cabecera}  \n_Términos:_ {terminos}")
        self.m["nombres"].markdown("\n\n".join(lineas))


def esperar(segundos: float) -> None:
    """Pausa de la reproducción (punto único que las pruebas sustituyen)."""
    time.sleep(segundos)


def reproducir(escena: Escena, eventos: Iterable[Evento], velocidad: str, en_vivo: bool, dormir=None) -> None:
    """Muestra los eventos al ritmo de ``velocidad``. En vivo, ``eventos`` es el generador que calcula de verdad."""
    dormir = dormir or esperar
    for ev in eventos:
        pausa = espera(ev, velocidad, en_vivo)
        if pausa > 0:
            dormir(pausa)
        escena.mostrar(ev)
