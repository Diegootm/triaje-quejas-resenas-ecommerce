"""Concurrencia del almacén de la app: un lock por K y el barrido sin bloqueo global."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import estado  # noqa: E402
from codigo import agrupamiento  # noqa: E402

ESPERA = 10  # segundos máximos para que un hilo termine (si se bloquea, la prueba falla)


@pytest.fixture
def base():
    return SimpleNamespace(
        k_referencia=6, kmeans_referencia="modelo-guardado",
        X_svd_corpus=np.zeros((10, 3)),
        con_kmeans=lambda modelo: SimpleNamespace(modelo=modelo),
    )


def _en_hilo(funcion, *args):
    resultado = {}
    hilo = threading.Thread(target=lambda: resultado.setdefault("valor", funcion(*args)), daemon=True)
    hilo.start()
    return hilo, resultado


def test_el_k_guardado_no_entrena_nada(base, monkeypatch):
    monkeypatch.setattr(agrupamiento, "entrenar_kmeans", lambda *a, **k: pytest.fail("no debe entrenar"))
    almacen = estado.Almacen()
    assert almacen.modelo_para_k(base, 6) == "modelo-guardado"
    assert almacen.agente_para_k(base, 6) is base and almacen.agente_listo(base, 6)


def test_sesiones_simultaneas_con_el_mismo_k_entrenan_una_sola_vez(base, monkeypatch):
    llamadas = []

    def lento(X, k):
        llamadas.append(k)
        time.sleep(0.2)
        return f"modelo-{k}"

    monkeypatch.setattr(agrupamiento, "entrenar_kmeans", lento)
    almacen = estado.Almacen()
    hilos = [_en_hilo(almacen.agente_para_k, base, 8) for _ in range(6)]
    for hilo, _ in hilos:
        hilo.join(ESPERA)
    assert llamadas == [8]
    assert len({id(r["valor"]) for _, r in hilos}) == 1 and almacen.agente_listo(base, 8)


def test_entrenar_un_k_no_bloquea_a_otro_k(base, monkeypatch):
    monkeypatch.setattr(agrupamiento, "entrenar_kmeans", lambda X, k: f"modelo-{k}")
    almacen = estado.Almacen()
    with almacen._lock_de(5):                       # otra sesión está entrenando K = 5
        hilo, resultado = _en_hilo(almacen.modelo_para_k, base, 7)
        hilo.join(ESPERA)
        assert not hilo.is_alive() and resultado["valor"] == "modelo-7"


def test_el_barrido_no_bloquea_el_entrenamiento_de_un_k(base, monkeypatch):
    liberar = threading.Event()
    tabla = pd.DataFrame({"K": [2, 3]})

    def barrido_lento(X, rango, progreso=None):
        liberar.wait(ESPERA)
        return tabla, {2: "m2", 3: "m3"}

    monkeypatch.setattr(agrupamiento, "barrido_k", barrido_lento)
    monkeypatch.setattr(agrupamiento, "entrenar_kmeans", lambda X, k: f"modelo-{k}")
    almacen = estado.Almacen()
    hilo_barrido, _ = _en_hilo(almacen.calcular_barrido, base)
    time.sleep(0.1)
    hilo_k, resultado = _en_hilo(almacen.modelo_para_k, base, 9)
    hilo_k.join(ESPERA)
    assert not hilo_k.is_alive() and resultado["valor"] == "modelo-9"   # el barrido sigue en curso
    assert hilo_barrido.is_alive()
    liberar.set()
    hilo_barrido.join(ESPERA)
    assert almacen.barrido is tabla and almacen.modelos[2] == "m2" and almacen.modelos[9] == "modelo-9"


def test_el_barrido_se_calcula_una_sola_vez(base, monkeypatch):
    llamadas = []
    tabla = pd.DataFrame({"K": [2]})
    monkeypatch.setattr(agrupamiento, "barrido_k", lambda X, rango, progreso=None: (llamadas.append(1), (tabla, {2: "m2"}))[1])
    almacen = estado.Almacen()
    assert almacen.calcular_barrido(base) is almacen.calcular_barrido(base)
    assert len(llamadas) == 1
