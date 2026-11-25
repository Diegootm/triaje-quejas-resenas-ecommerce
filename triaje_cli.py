"""Triaje desde la terminal (alternativa rápida a la interfaz Streamlit).

    python triaje_cli.py "It runs very small, I had to return it for a larger size."
    python triaje_cli.py            # modo interactivo: escribe quejas, Enter vacío para salir
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from codigo import agrupamiento, config  # noqa: E402
from codigo.decision_k import ORIGEN_MANUAL, cargar_decision  # noqa: E402
from codigo.presentacion import numero_grupo  # noqa: E402
from codigo.preprocesamiento import RecursoNltkFaltante, verificar_recursos_nltk  # noqa: E402
from codigo.triaje import AgenteTriaje  # noqa: E402


def cargar_agente_con_k_del_agente() -> AgenteTriaje:
    """Carga el agente y usa el K que eligió al entrenar (modelos/seleccion_k.json)."""
    decision = cargar_decision(config.DIR_MODELOS)
    base = AgenteTriaje()
    agente = base
    if decision.k_elegido != base.k_referencia:
        agente = base.con_kmeans(agrupamiento.entrenar_kmeans(base.X_svd_corpus, k=decision.k_elegido))
    origen = "impuesto con --k" if decision.origen == ORIGEN_MANUAL else "elegido por el agente"
    print(f"Agente de triaje con K = {agente.n_grupos} ({origen}).")
    return agente


def mostrar(agente: AgenteTriaje, queja: str) -> None:
    r = agente.sugerir(queja)
    print("\nQueja:", r.queja)
    print("Texto preprocesado:", r.texto_limpio or "(vacío)")
    if not r.valido:
        print("AVISO:", r.aviso)
        return
    print("Distancias a los grupos:", " / ".join(f"{numero_grupo(i)}:{d:.4f}" for i, d in enumerate(r.distancias)))
    print(f"GRUPO SUGERIDO: {numero_grupo(r.grupo)} — {r.nombre}")
    if r.nombre_informe:
        print(f"Nombre del informe: {r.nombre_informe}")
    print("Términos del grupo:", ", ".join(r.terminos[:8]))
    if r.aviso:
        print("AVISO:", r.aviso)
    print("Reseñas más parecidas:")
    for s in r.similares:
        print(f"  - [coseno={s['similitud']:.3f}, grupo {numero_grupo(s['grupo'])}] {s['texto'][:150]}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        verificar_recursos_nltk()
        agente = cargar_agente_con_k_del_agente()
    except (RecursoNltkFaltante, FileNotFoundError) as error:
        sys.exit(f"error: {error}")
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
