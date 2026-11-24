"""Humo de la app Streamlit con ``AppTest`` (sin navegador) y del triaje por terminal.

Las pruebas que necesitan el agente se omiten si no hay modelos entrenados.
``AppTest`` (Streamlit 1.40) no puede subir archivos, así que el lote CSV se prueba por su
función ``clasificar_lote`` y la pestaña se comprueba solo hasta "sin archivo cargado".
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from codigo import config  # noqa: E402

SCRIPT = str(RAIZ / "app" / "app_streamlit.py")
HAY_MODELOS = (RAIZ / "modelos" / "modelo_kmeans.joblib").exists() and (RAIZ / "modelos" / config.ARCHIVO_DECISION_K).exists()
NECESITA_MODELOS = pytest.mark.skipif(not HAY_MODELOS, reason="Primero entrena: python entrenar_agente.py")
QUEJA = "It runs very small, I had to return it for a larger size."


def _app() -> AppTest:
    return AppTest.from_file(SCRIPT, default_timeout=180)


def _textos(elementos) -> str:
    return "\n".join(str(e.value) for e in elementos)


def _sugerir(at: AppTest, texto: str = QUEJA) -> AppTest:
    at.text_area(key="queja").set_value(texto)
    next(b for b in at.button if b.label == "Sugerir grupo").click()
    return at.run()


# --------------------------------------------------------------------------
# Errores claros, sin traceback (no necesitan modelos)
# --------------------------------------------------------------------------
def test_sin_recursos_nltk_muestra_el_comando_y_no_falla(monkeypatch):
    monkeypatch.setattr("app.estado.recursos_nltk_faltantes", lambda: ["wordnet"])
    at = _app().run()
    assert not at.exception
    assert "wordnet" in _textos(at.error)
    assert "python preparar_entorno.py" in _textos(at.code)


def test_sin_decision_de_k_pide_entrenar_y_no_falla(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DIR_MODELOS", tmp_path)
    at = _app().run()
    assert not at.exception
    assert "entrena con python entrenar_agente.py" in _textos(at.error)
    assert "python entrenar_agente.py" in _textos(at.code)


# --------------------------------------------------------------------------
# Con modelos entrenados
# --------------------------------------------------------------------------
@NECESITA_MODELOS
def test_arranca_con_el_k_del_agente():
    from codigo.decision_k import cargar_decision
    k = cargar_decision(config.DIR_MODELOS).k_elegido
    at = _app().run()
    assert not at.exception
    lateral = _textos(at.sidebar.markdown)
    assert f"K elegido por el agente: {k}" in lateral and "K manual" not in lateral
    assert at.sidebar.slider[0].value == k
    assert [t.label for t in at.tabs] == ["Queja individual", "Lote CSV", "Grupos descubiertos", "Selección de K"]


@NECESITA_MODELOS
def test_queja_individual_usa_el_k_del_agente():
    from codigo.decision_k import cargar_decision
    k = cargar_decision(config.DIR_MODELOS).k_elegido
    at = _sugerir(_app().run())
    assert not at.exception
    assert f"Grupo sugerido (K = {k})" in _textos(at.success)


@NECESITA_MODELOS
def test_k_manual_cambia_todas_las_pestanas_y_se_puede_volver_al_del_agente():
    from codigo.decision_k import cargar_decision
    k_agente = cargar_decision(config.DIR_MODELOS).k_elegido
    k_manual = 8 if k_agente != 8 else 7
    at = _app().run()
    at.sidebar.slider[0].set_value(k_manual).run()
    assert not at.exception
    lateral = _textos(at.sidebar.markdown)
    assert f"K manual: {k_manual}" in lateral and f"K elegido por el agente: {k_agente}" in lateral
    assert f"K = {k_manual}" in _textos(at.subheader)           # pestaña "Grupos descubiertos"
    at = _sugerir(at)
    assert f"Grupo sugerido (K = {k_manual})" in _textos(at.success)

    next(b for b in at.sidebar.button if b.label.startswith("Volver al K del agente")).click()
    at.run()
    assert at.sidebar.slider[0].value == k_agente
    assert "K manual" not in _textos(at.sidebar.markdown)
    assert f"Grupo sugerido (K = {k_agente})" in _textos(_sugerir(at).success)


@NECESITA_MODELOS
def test_lote_csv_muestra_el_selector_de_archivo_y_clasifica_con_el_k_activo():
    from app.vista_triaje import clasificar_lote
    from codigo.triaje import AgenteTriaje
    at = _app().run()
    assert not at.exception
    assert "Todavía no se ha cargado ningún archivo" in _textos(at.caption)

    agente = AgenteTriaje()
    avances = []
    resultado = clasificar_lote(agente, [QUEJA, "asdf qwerty"], progreso=avances.append)
    assert resultado["K"].tolist() == [agente.n_grupos] * 2
    assert resultado["grupo"].iloc[0] == 4 and resultado["nombre"].iloc[1] == "Sin vocabulario conocido"
    assert avances == [0.5, 1.0]


@NECESITA_MODELOS
def test_triaje_cli_usa_el_k_del_agente():
    from codigo.decision_k import cargar_decision
    k = cargar_decision(config.DIR_MODELOS).k_elegido
    salida = subprocess.run(
        [sys.executable, str(RAIZ / "triaje_cli.py"), QUEJA],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180,
    )
    assert salida.returncode == 0, salida.stderr
    assert f"K = {k} (elegido por el agente)" in salida.stdout and "GRUPO SUGERIDO" in salida.stdout
