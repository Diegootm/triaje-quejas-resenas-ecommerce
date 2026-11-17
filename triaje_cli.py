"""Triaje desde la terminal (alternativa rápida a la interfaz Streamlit).

    python triaje_cli.py "It runs very small, I had to return it for a larger size."
    python triaje_cli.py            # modo interactivo: escribe quejas, Enter vacío para salir
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from codigo.triaje import AgenteTriaje  # noqa: E402


def mostrar(agente: AgenteTriaje, queja: str) -> None:
    r = agente.sugerir(queja)
    print("\nQueja:", r.queja)
    print("Texto preprocesado:", r.texto_limpio or "(vacío)")
    if not r.valido:
        print("AVISO:", r.aviso)
        return
    print("Distancias a los grupos:", " / ".join(f"{i}:{d:.4f}" for i, d in enumerate(r.distancias)))
    print(f"GRUPO SUGERIDO: {r.grupo} — {r.nombre}")
    print("Términos del grupo:", ", ".join(r.terminos[:8]))
    if r.aviso:
        print("AVISO:", r.aviso)
    print("Reseñas más parecidas:")
    for s in r.similares:
        print(f"  - [coseno={s['similitud']:.3f}, grupo {s['grupo']}] {s['texto'][:150]}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    agente = AgenteTriaje()
    if len(sys.argv) > 1:
        for queja in sys.argv[1:]:
            mostrar(agente, queja)
        return
    print("Escribe una queja en inglés (Enter vacío para salir).")
    while True:
        queja = input("\n> ").strip()
        if not queja:
            break
        mostrar(agente, queja)


if __name__ == "__main__":
    main()
