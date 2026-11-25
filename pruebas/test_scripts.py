"""Los scripts de la raíz se pueden ejecutar (detecta errores de sintaxis o de importación)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("script", ["entrenar_agente.py", "triaje_cli.py"])
def test_script_compila(script):
    compile((RAIZ / script).read_text(encoding="utf-8"), script, "exec")


def test_entrenar_agente_muestra_la_ayuda():
    resultado = subprocess.run(
        [sys.executable, str(RAIZ / "entrenar_agente.py"), "--help"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
    )
    assert resultado.returncode == 0, resultado.stderr
    assert "--k" in resultado.stdout
