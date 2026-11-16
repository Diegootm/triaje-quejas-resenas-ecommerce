"""Punto de entrada de la interfaz Streamlit del proyecto.

Este archivo es un esqueleto funcional: permite probar la pantalla y el flujo
de entrada, pero todavía NO carga modelos entrenados ni realiza inferencia real.
Cuando exista un modelo en ``modelos/``, se conectará aquí.

Ejecución local:

    streamlit run app/app_streamlit.py
"""

from __future__ import annotations

import streamlit as st

DESCRIPCION = (
    "Prototipo para agrupar y priorizar quejas a partir de reseñas de "
    "e-commerce. El texto de las reseñas se mantiene en su idioma original "
    "(inglés)."
)

ESTADO = (
    "La base del repositorio está creada. Aún no se han entrenado modelos ni "
    "ejecutado pruebas, por lo que esta pantalla no produce clasificaciones "
    "reales."
)


def _configurar_pagina() -> None:
    st.set_page_config(
        page_title="Triaje de quejas de reseñas",
        page_icon=":material/space_dashboard:",
        layout="wide",
    )


def _barra_lateral() -> None:
    with st.sidebar:
        st.header("Configuración")
        st.caption(
            "La selección de modelo se habilitará cuando exista un artefacto "
            "entrenado en la carpeta modelos/."
        )
        st.selectbox(
            "Modelo",
            options=["Sin modelo entrenado"],
            disabled=True,
            help="Pendiente: cargar el modelo y el vectorizador TF-IDF.",
        )
        st.selectbox(
            "Definición de queja",
            options=[
                "Pendiente de definir (Rating / Recommended IND)",
            ],
            disabled=True,
            help="La definición se documentará y se fijará en una etapa posterior.",
        )
        st.divider()
        st.markdown("**Estado**")
        st.info(ESTADO)


def _pestana_texto() -> None:
    st.subheader("Analizar una reseña")
    texto = st.text_area(
        "Texto de la reseña (inglés)",
        height=180,
        placeholder="Ejemplo: The fabric felt cheap and the size ran small...",
    )
    if st.button("Analizar reseña", type="primary"):
        if not texto.strip():
            st.warning("Escribe el texto de una reseña para continuar.")
        else:
            st.info(
                "Entrada recibida. La clasificación estará disponible cuando se "
                "entrene y cargue el modelo."
            )


def _pestana_lote() -> None:
    st.subheader("Analizar un lote de reseñas")
    archivo = st.file_uploader(
        "Archivo CSV con reseñas",
        type=["csv"],
        help="Se espera una columna de texto, por ejemplo 'Review Text'.",
    )
    if archivo is not None:
        st.success(f"Archivo cargado: {archivo.name}")
        st.caption(
            "La lectura, validación de columnas y clasificación por lote se "
            "implementarán junto con el pipeline de preprocesamiento."
        )
    else:
        st.caption("Todavía no se ha cargado ningún archivo.")


def main() -> None:
    """Renderiza la aplicación."""
    _configurar_pagina()

    st.title("Triaje de quejas de reseñas de e-commerce")
    st.write(DESCRIPCION)

    _barra_lateral()

    pestana_texto, pestana_lote, pestana_estado = st.tabs(
        ["Reseña individual", "Lote CSV", "Estado del proyecto"]
    )

    with pestana_texto:
        _pestana_texto()

    with pestana_lote:
        _pestana_lote()

    with pestana_estado:
        st.subheader("Alcance actual")
        st.markdown(
            "- Estructura del repositorio y módulos documentados.\n"
            "- Interfaz mínima sin dependencia de modelos entrenados.\n"
            "- Pendiente: preprocesamiento, TF-IDF, KMeans, evaluación y "
            "etiquetado.\n"
            "- Pendiente: definición formal de \"queja\" y su justificación."
        )
        st.caption(
            "Los datos crudos pesados y los modelos generados no se versionan; "
            "consulta el README para preparar el entorno."
        )


if __name__ == "__main__":
    main()
