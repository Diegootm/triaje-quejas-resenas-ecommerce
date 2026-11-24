"""Pruebas de la verificación explícita de recursos de NLTK (sin descargas)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codigo import preprocesamiento  # noqa: E402


@pytest.fixture
def sin_recursos(monkeypatch):
    """Simula un entorno sin recursos de NLTK y sin permitir descargas."""
    def no_encontrado(ruta):
        raise LookupError(ruta)

    def no_descargar(*args, **kwargs):
        raise AssertionError("no debe descargar nada dentro de las funciones")

    monkeypatch.setattr(preprocesamiento.nltk.data, "find", no_encontrado)
    monkeypatch.setattr(preprocesamiento.nltk, "download", no_descargar)
    monkeypatch.setattr(preprocesamiento, "_recursos_verificados", False)


def test_faltantes_lista_todos_los_recursos(sin_recursos):
    assert preprocesamiento.recursos_nltk_faltantes() == list(preprocesamiento.RECURSOS_NLTK)


def test_mensaje_indica_como_instalar(sin_recursos):
    with pytest.raises(preprocesamiento.RecursoNltkFaltante) as error:
        preprocesamiento.verificar_recursos_nltk()
    assert preprocesamiento.COMANDO_PREPARAR_ENTORNO in str(error.value)
    assert "stopwords" in str(error.value)


@pytest.mark.parametrize("llamada", [
    lambda: preprocesamiento.tokenizar("not good"),
    lambda: preprocesamiento.lematizar(["dresses"]),
])
def test_funciones_fallan_claro_sin_descargar(sin_recursos, llamada):
    with pytest.raises(preprocesamiento.RecursoNltkFaltante):
        llamada()


def test_con_recursos_instalados_no_falla():
    assert preprocesamiento.recursos_nltk_faltantes() == []
    preprocesamiento.verificar_recursos_nltk()
