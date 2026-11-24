"""Descarga, de forma visible, los recursos de NLTK que necesita el proyecto.

Ejecutar una sola vez después de instalar las dependencias:  python preparar_entorno.py
"""

from __future__ import annotations

import sys

import nltk

from codigo.preprocesamiento import RECURSOS_NLTK, recursos_nltk_faltantes


def main() -> int:
    pendientes = recursos_nltk_faltantes()
    for nombre in RECURSOS_NLTK:
        if nombre not in pendientes:
            print(f"[ya instalado] {nombre}")
            continue
        print(f"[descargando]  {nombre} ...", flush=True)
        if not nltk.download(nombre, quiet=True):
            print(f"[error]        no se pudo descargar {nombre} (revisa tu conexión)")

    faltantes = recursos_nltk_faltantes()
    if faltantes:
        print(f"\nFaltan: {', '.join(faltantes)}. Vuelve a ejecutar este script.")
        return 1
    print("\nEntorno listo: todos los recursos de NLTK están instalados.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
