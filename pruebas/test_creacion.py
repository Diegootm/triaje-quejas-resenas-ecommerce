"""Eventos de la creación de categorías: orden, datos reales, comprobación contra el guardado y ritmo de reproducción."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from app import escena_creacion  # noqa: E402
from codigo import agrupamiento, config, creacion  # noqa: E402
from codigo.decision_k import decidir_k  # noqa: E402
from codigo.evaluacion import indice_rand_ajustado  # noqa: E402

K = 4


@pytest.fixture(scope="module")
def mundo():
    """Un agente mínimo (mismos atributos que usa creacion) sobre datos sintéticos, y su decisión de K."""
    X, _ = make_blobs(n_samples=600, centers=K, n_features=8, cluster_std=2.0, random_state=5)
    modelo = KMeans(n_clusters=K, random_state=config.SEMILLA, n_init=config.N_INIT).fit(X)
    tabla, _ = agrupamiento.barrido_k(X)
    etiquetas = modelo.labels_
    base = SimpleNamespace(
        X_svd_corpus=X, k_referencia=K, kmeans_referencia=modelo, etiquetas_referencia=etiquetas,
        corpus=pd.DataFrame({"Review Text": [f"reseña número {i} " * 12 for i in range(len(X))]}),
        terminos={g: [f"t{g}{i}" for i in range(config.N_TERMINOS_TOP)] for g in range(K)},
        nombres={g: f"Grupo {g + 1}: t{g}0, t{g}1, t{g}2" for g in range(K)}, nombres_secundarios={0: "Nombre del informe"},
    )
    return base, decidir_k(tabla), list(creacion.generar_eventos(base, decidir_k(tabla)))


def _tipos(eventos):
    return [e.tipo for e in eventos]


def test_el_orden_de_las_fases_es_el_de_la_pantalla(mundo):
    _, _, eventos = mundo
    assert [e.fase for e in eventos] == sorted(e.fase for e in eventos)
    tipos = Counter(_tipos(eventos))
    assert tipos["mapa"] == 1 and tipos["k_listo"] == tipos["k_probando"] == len(config.RANGO_K)
    assert tipos["arranque_listo"] == config.N_INIT and tipos["semilla"] == K and tipos["nombre_listo"] == K
    assert _tipos(eventos)[0] == "mapa" and _tipos(eventos)[-1] == "fin"
    assert _tipos(eventos).index("k_decidido") > max(i for i, t in enumerate(_tipos(eventos)) if t == "k_listo")
    assert _tipos(eventos).index("ganador") > max(i for i, t in enumerate(_tipos(eventos)) if t == "arranque_listo")


def test_cada_evento_se_puede_narrar_y_ninguna_duracion_es_negativa(mundo):
    _, _, eventos = mundo
    assert all(creacion.narracion(e) and e.duracion >= 0 for e in eventos)
    assert creacion.narracion(next(e for e in eventos if e.tipo == "k_probando")) == "Probando 2 categorías…"


def test_la_fase_1_repite_la_tabla_guardada_y_la_misma_decision(mundo):
    _, decision, eventos = mundo
    decidido = next(e for e in eventos if e.tipo == "k_decidido").datos
    assert decidido["comprobacion"]["coincide"] and decidido["k"] == decision.k_elegido
    assert decidido["umbral"] == pytest.approx(decision.umbral) and "Elegí K=" in decidido["explicacion"]
    for e in (e for e in eventos if e.tipo == "k_listo"):
        assert len(e.datos["etiquetas"]) == 600 and e.datos["etiquetas"].max() == e.datos["K"] - 1


def test_comparar_con_guardado_detecta_una_tabla_distinta(mundo):
    _, decision, _ = mundo
    alterada = decision.tabla.assign(silueta=decision.tabla["silueta"] * 1.5)
    assert not creacion.comparar_con_guardado(alterada, decision)["coincide"]
    assert creacion.comparar_con_guardado(decision.tabla, decision)["tabla_igual"]


def test_la_ganadora_reproduce_el_modelo_guardado(mundo):
    base, _, eventos = mundo
    corridas = [e.datos for e in eventos if e.tipo == "arranque_listo"]
    ganador = next(e for e in eventos if e.tipo == "ganador").datos
    assert ganador["inercia"] == min(c["inercia"] for c in corridas)
    assert ganador["inercia"] == pytest.approx(base.kmeans_referencia.inertia_)
    final = [e for e in eventos if e.tipo == "iteracion"][-1].datos
    assert indice_rand_ajustado(final["etiquetas"], base.etiquetas_referencia) == 1.0
    assert next(e for e in eventos if e.tipo == "estable").datos["ari"] == 1.0


def test_las_iteraciones_parten_de_las_semillas_y_la_inercia_no_sube(mundo):
    _, _, eventos = mundo
    semillas = np.array([e.datos["centro2d"] for e in eventos if e.tipo == "semilla"])
    iteraciones = [e.datos for e in eventos if e.tipo == "iteracion"]
    assert iteraciones[0]["iteracion"] == 0 and iteraciones[0]["cambios"] == 0
    assert np.allclose(iteraciones[0]["centros2d"], semillas)
    inercias = [d["inercia"] for d in iteraciones]
    assert all(b <= a + 1e-9 for a, b in zip(inercias, inercias[1:]))
    assert [d["iteracion"] for d in iteraciones] == list(range(len(iteraciones)))


def test_la_fase_3_entrega_terminos_en_orden_y_el_conteo_llega_al_total(mundo):
    base, _, eventos = mundo
    g0 = [e.datos for e in eventos if e.fase == 3 and e.datos.get("g") == 0]
    assert [d["termino"] for d in g0 if "termino" in d] == base.terminos[0]
    conteos = [d["n"] for d in g0 if "n" in d]
    assert len(conteos) == config.PASOS_CONTEO and conteos == sorted(conteos)
    assert conteos[-1] == int((base.etiquetas_referencia == 0).sum())
    nombres = {e.datos["g"]: e.datos["nombre_informe"] for e in eventos if e.tipo == "nombre_listo"}
    assert nombres[0] == "Nombre del informe" and nombres[1] is None


def test_la_duracion_por_fase_suma_lo_calculado(mundo):
    _, _, eventos = mundo
    por_fase = creacion.duracion_por_fase(eventos)
    assert set(por_fase) == {0, 1, 2, 3} and por_fase[1] > 0 and por_fase[2] > 0
    assert por_fase[1] == pytest.approx(sum(e.duracion for e in eventos if e.fase == 1))


# --------------------------------------------------------------------------
# Ritmo de reproducción
# --------------------------------------------------------------------------
PASO = creacion.Evento(2, "iteracion", {}, duracion=0.3)
DETALLE = creacion.Evento(3, "termino", {}, duracion=0.0, detalle=True)


def test_velocidad_real_espera_lo_que_tardo_el_calculo_solo_al_repetir():
    assert creacion.espera(PASO, "Velocidad real", en_vivo=False) == 0.3
    assert creacion.espera(PASO, "Velocidad real", en_vivo=True) == 0.0       # en vivo ya se esperó calculando
    assert creacion.espera(DETALLE, "Velocidad real", en_vivo=False) == 0.0


def test_velocidad_lenta_es_una_pausa_fija_que_descuenta_lo_ya_calculado_en_vivo():
    paso, detalle = config.VELOCIDADES["Lenta"]
    assert creacion.espera(PASO, "Lenta", en_vivo=False) == paso
    assert creacion.espera(PASO, "Lenta", en_vivo=True) == pytest.approx(max(0.0, paso - 0.3))
    assert creacion.espera(DETALLE, "Lenta", en_vivo=False) == detalle


def test_las_velocidades_lentas_son_cada_vez_mas_lentas():
    pasos = [v[0] for n, v in config.VELOCIDADES.items() if n != "Velocidad real"]
    assert pasos == sorted(pasos) and config.VELOCIDADES["Velocidad real"] == (None, None)
    assert config.VELOCIDAD_INICIAL in config.VELOCIDADES


def test_fragmentos_cortan_escapan_y_parten_en_lineas():
    largo = "word " * 100
    frag = creacion.fragmentos_resenas([largo, "a <b> & c", ""])
    assert len(frag[0].replace("<br>", " ")) <= config.LARGO_FRAGMENTO_RESENA + 3 and frag[0].endswith("…")
    assert "<br>" in frag[0] and frag[1] == "a &lt;b&gt; &amp; c" and frag[2] == ""


def test_reproducir_duerme_lo_que_dice_espera_y_muestra_todo_en_orden():
    pausas, vistos = [], []
    escena = SimpleNamespace(mostrar=lambda ev: vistos.append(ev.tipo))
    eventos = [PASO, DETALLE, PASO]
    escena_creacion.reproducir(escena, eventos, "Lenta", en_vivo=False, dormir=pausas.append)
    paso, detalle = config.VELOCIDADES["Lenta"]
    assert vistos == ["iteracion", "termino", "iteracion"] and pausas == [paso, detalle, paso]
    pausas.clear()
    escena_creacion.reproducir(escena, eventos, "Velocidad real", en_vivo=True, dormir=pausas.append)
    assert pausas == []


# --------------------------------------------------------------------------
# Submuestra fija del dibujo (los cálculos usan todo el corpus)
# --------------------------------------------------------------------------
def test_la_submuestra_es_fija_ordenada_sin_repetidos_y_del_tamano_configurado(mundo, monkeypatch):
    base, decision, _ = mundo
    monkeypatch.setattr(config, "SUBMUESTRA_MAPA", 100)
    a, b = creacion.preparar_mapa(base, decision), creacion.preparar_mapa(base, decision)
    assert len(a.indices) == 100 and list(a.indices) == sorted(set(a.indices)) and a.es_submuestra
    assert list(a.indices) == list(b.indices) and a.indices.max() < 600
    assert a.puntos2d.shape == (100, 2) and len(a.fragmentos_dibujo) == 100
    assert np.allclose(a.puntos2d, a.proyeccion.puntos2d[a.indices])


def test_si_la_muestra_cubre_todo_el_corpus_no_se_declara_submuestra(mundo):
    base, decision, _ = mundo
    mapa = creacion.preparar_mapa(base, decision)           # 600 quejas < SUBMUESTRA_MAPA
    assert not mapa.es_submuestra and len(mapa.indices) == 600


def test_la_escena_dibuja_solo_la_muestra_pero_los_contadores_son_del_corpus_completo(mundo, monkeypatch):
    base, decision, eventos = mundo
    monkeypatch.setattr(config, "SUBMUESTRA_MAPA", 100)
    figuras = {}

    class Marco:
        def __init__(self, nombre): self.nombre = nombre
        def plotly_chart(self, fig, **kw): figuras[self.nombre] = fig
        def markdown(self, texto): figuras[self.nombre] = texto
        def caption(self, texto): figuras[self.nombre] = texto

    marcos = {n: Marco(n) for n in ("narracion", "silueta", "motivos", "decision", "mapa", "contadores", "tira", "nombres")}
    marcos["mosaico"] = {k: Marco(f"mini{k}") for k in config.RANGO_K}
    escena = escena_creacion.Escena(marcos, creacion.preparar_mapa(base, decision))
    for ev in eventos:
        if ev.tipo not in ("k_decidido", "fin"):          # k_decidido usa st.info; fin pinta el estado final
            escena.mostrar(ev)
    assert len(figuras["mapa"].data[0].x) == 100 and len(figuras["mini4"].data[0].x) == 100
    ultimo = [e for e in eventos if e.tipo == "iteracion"][-1].datos
    assert f"{ultimo['inercia']:,.2f}" in figuras["contadores"]
    assert f"de {int((base.etiquetas_referencia == 0).sum()):,} quejas" in figuras["nombres"]
