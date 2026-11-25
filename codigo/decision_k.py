"""Decisión de K del agente: qué K eligió, con qué evidencia y por qué.

La regla es la de :func:`codigo.agrupamiento.recomendar_k`. Este módulo la
envuelve en un objeto serializable (``modelos/seleccion_k.json``) para que la
interfaz, la terminal y las pruebas lean la MISMA decisión sin recalcularla.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import pandas as pd

from codigo.agrupamiento import recomendar_k
from codigo.config import ARCHIVO_DECISION_K, SEMILLA, TOLERANCIA_SILUETA

ORIGEN_AGENTE = "agente"
ORIGEN_MANUAL = "manual"


@dataclass(frozen=True)
class DecisionK:
    """Resultado de elegir K. ``origen`` es ``"agente"`` o ``"manual"`` (``--k N``)."""

    tabla: pd.DataFrame          # barrido completo: K, inercia, silueta, tamaños
    descartados: dict[int, str]  # K descartados y su motivo
    k_elegido: int
    k_mejor_silueta: int
    silueta_mejor: float         # s*
    umbral: float                # s* - tolerancia * |s*|
    margen: float                # silueta del K elegido - umbral
    tolerancia: float
    semilla: int
    fecha: str
    origen: str = ORIGEN_AGENTE

    @property
    def candidatos(self) -> list[int]:
        return [int(k) for k in self.tabla["K"] if int(k) not in self.descartados]

    def silueta_de(self, k: int) -> float:
        return float(self.tabla.set_index("K").loc[k, "silueta"])


def decidir_k(tabla: pd.DataFrame, tolerancia: float = TOLERANCIA_SILUETA, semilla: int = SEMILLA) -> DecisionK:
    """Aplica la regla de K a la tabla del barrido. Falla si ningún K cumple los filtros."""
    rec = recomendar_k(tabla, tolerancia=tolerancia)
    if rec.k is None:
        raise ValueError("Ningún K cumple los filtros de la regla: " + "; ".join(
            f"K={k}: {motivo}" for k, motivo in rec.descartados.items()))
    silueta_elegida = float(tabla.set_index("K").loc[rec.k, "silueta"])
    return DecisionK(
        tabla=tabla.sort_values("K").reset_index(drop=True),
        descartados=dict(rec.descartados),
        k_elegido=rec.k,
        k_mejor_silueta=rec.k_mejor_silueta,
        silueta_mejor=rec.silueta_mejor,
        umbral=rec.umbral_silueta,
        margen=silueta_elegida - rec.umbral_silueta,
        tolerancia=tolerancia,
        semilla=semilla,
        fecha=datetime.now().isoformat(timespec="seconds"),
    )


def con_k_manual(decision: DecisionK, k: int) -> DecisionK:
    """La misma evidencia, pero con un K impuesto por la persona (``--k N``)."""
    if k not in set(decision.tabla["K"]):
        raise ValueError(f"K={k} no está en la tabla del barrido")
    return replace(decision, k_elegido=int(k), margen=decision.silueta_de(k) - decision.umbral, origen=ORIGEN_MANUAL)


def guardar_decision(decision: DecisionK, directorio: str | Path) -> Path:
    directorio = Path(directorio)
    directorio.mkdir(parents=True, exist_ok=True)
    datos = {
        **{k: v for k, v in decision.__dict__.items() if k not in ("tabla", "descartados")},
        "descartados": {str(k): v for k, v in decision.descartados.items()},
        "tabla": decision.tabla.to_dict(orient="records"),
    }
    ruta = directorio / ARCHIVO_DECISION_K
    ruta.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")
    return ruta


def cargar_decision(directorio: str | Path) -> DecisionK:
    ruta = Path(directorio) / ARCHIVO_DECISION_K
    if not ruta.exists():
        raise FileNotFoundError(f"No existe {ruta}: entrena con python entrenar_agente.py")
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    datos["tabla"] = pd.DataFrame(datos["tabla"])
    datos["descartados"] = {int(k): v for k, v in datos["descartados"].items()}
    return DecisionK(**datos)


def explicar_decision(d: DecisionK) -> str:
    """Texto "Elegí K=X porque…" con las cifras reales de la decisión."""
    partes = []
    if d.descartados:
        motivos = "; ".join(f"K={k} ({motivo})" for k, motivo in sorted(d.descartados.items()))
        partes.append(f"Descarté {motivos}.")
    cand = d.candidatos
    partes.append(
        f"Entre los candidatos (K={cand[0]} a K={cand[-1]}) la mejor silueta es s*={d.silueta_mejor:.4f} "
        f"(K={d.k_mejor_silueta}). Acepto hasta {d.tolerancia:.0%} por debajo: "
        f"umbral = {d.silueta_mejor:.4f} - {d.tolerancia:.2f}·{abs(d.silueta_mejor):.4f} = {d.umbral:.4f}.")
    silueta = d.silueta_de(d.k_elegido)
    if d.origen == ORIGEN_AGENTE:
        partes.insert(0, f"Elegí K={d.k_elegido}.")
        partes.append(
            f"K={d.k_elegido} es el K más pequeño que alcanza el umbral (silueta {silueta:.4f}, "
            f"{d.margen:+.4f} sobre el umbral): menos grupos con calidad comparable.")
    else:
        partes.insert(0, f"K={d.k_elegido} fue impuesto manualmente (--k); no lo eligió el agente.")
        partes.append(f"Con K={d.k_elegido} la silueta es {silueta:.4f}, {d.margen:+.4f} respecto del umbral.")
    return " ".join(partes)
