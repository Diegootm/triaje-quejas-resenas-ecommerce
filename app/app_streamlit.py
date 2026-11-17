"""Interfaz Streamlit del agente de triaje de quejas.

Carga los modelos entrenados (vectorizador TF-IDF, SVD y K-means) y, para cada
queja nueva, sugiere el grupo más cercano junto con sus términos
representativos y las reseñas más parecidas del corpus.

Ejecución local (desde la raíz del repositorio):

    streamlit run app/app_streamlit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from codigo.triaje import AgenteTriaje  # noqa: E402

EJEMPLOS = {
    "Talla": "It runs very small, I had to return it for a larger size.",
    "Calidad y precio": "Poor quality fabric, not worth the price.",
    "Silueta": "The cut is not flattering at all, too many pleats.",
    "Apariencia": "It looked nothing like the picture, very boxy and wide.",
}


def _configurar_pagina() -> None:
    st.set_page_config(page_title="Triaje de quejas de reseñas", page_icon=":material/space_dashboard:", layout="wide")


@st.cache_resource(show_spinner="Cargando modelos entrenados...")
def _cargar_agente() -> AgenteTriaje:
    return AgenteTriaje()


def _barra_lateral(agente: AgenteTriaje) -> None:
    with st.sidebar:
        st.header("Modelo cargado")
        st.markdown(
            f"- Vocabulario TF-IDF: **{len(agente.vectorizador.vocabulary_):,}** términos (1-2 gramas)\n"
            f"- SVD truncada: **{agente.svd.n_components}** componentes\n"
            f"- K-means: **K = {agente.n_grupos}** grupos\n"
            f"- Corpus de quejas: **{len(agente.corpus):,}** reseñas"
        )
        st.divider()
        st.markdown("**Criterio de queja**")
        st.caption("Rating ≤ 2  **o**  Recommended IND = 0  **o**  VADER compound ≤ −0.05")
        st.divider()
        st.caption(
            "El agente solo **sugiere** un grupo: no resuelve la queja ni responde al cliente. "
            "El modelo fue entrenado con reseñas en **inglés**."
        )


def _mostrar_resultado(agente: AgenteTriaje, texto: str) -> None:
    r = agente.sugerir(texto, n_similares=5)
    st.markdown(f"**Texto preprocesado:** `{r.texto_limpio or '(vacío)'}`")
    if not r.valido:
        st.error(r.aviso)
        st.caption("Prueba escribir la queja en inglés, por ejemplo: " + EJEMPLOS["Talla"])
        return

    col1, col2 = st.columns([3, 2])
    with col1:
        st.success(f"**Grupo sugerido: {r.grupo} — {r.nombre}**")
        if r.aviso:
            st.warning(r.aviso)
        st.markdown("**Términos representativos del grupo:** " + ", ".join(f"`{t}`" for t in r.terminos[:8]))
    with col2:
        tabla = pd.DataFrame({
            "grupo": [f"{i}: {agente.nombres[i][:28]}" for i in range(agente.n_grupos)],
            "distancia": r.distancias,
        })
        st.markdown("**Distancia a cada centroide** (menor = más parecido)")
        st.bar_chart(tabla, x="grupo", y="distancia", horizontal=True, height=240)

    st.markdown("**Reseñas más parecidas del corpus** (similitud del coseno)")
    st.dataframe(
        pd.DataFrame(r.similares).rename(columns={"similitud": "coseno", "texto": "reseña"}),
        hide_index=True, use_container_width=True,
        column_config={"coseno": st.column_config.NumberColumn(format="%.3f")},
    )


def _pestana_texto(agente: AgenteTriaje) -> None:
    st.subheader("Analizar una queja")
    st.caption("Ejemplos rápidos:")
    columnas = st.columns(len(EJEMPLOS))
    for col, (etiqueta, ejemplo) in zip(columnas, EJEMPLOS.items()):
        if col.button(etiqueta, use_container_width=True):
            st.session_state["queja"] = ejemplo
    texto = st.text_area("Texto de la queja (inglés)", key="queja", height=120,
                         placeholder="Ejemplo: The fabric felt cheap and the size ran small...")
    if st.button("Sugerir grupo", type="primary"):
        if not texto.strip():
            st.warning("Escribe el texto de una queja para continuar.")
        else:
            _mostrar_resultado(agente, texto)


def _pestana_lote(agente: AgenteTriaje) -> None:
    st.subheader("Analizar un lote de quejas (CSV)")
    archivo = st.file_uploader("Archivo CSV", type=["csv"], help="Debe tener una columna de texto, p. ej. 'Review Text'.")
    if archivo is None:
        st.caption("Todavía no se ha cargado ningún archivo.")
        return
    df = pd.read_csv(archivo)
    columnas_texto = [c for c in df.columns if df[c].dtype == object]
    if not columnas_texto:
        st.error("El CSV no tiene columnas de texto.")
        return
    defecto = columnas_texto.index("Review Text") if "Review Text" in columnas_texto else 0
    columna = st.selectbox("Columna con el texto", columnas_texto, index=defecto)
    limite = st.number_input("Máximo de filas a procesar", 1, len(df), min(len(df), 500))
    if st.button("Clasificar lote", type="primary"):
        filas = []
        barra = st.progress(0.0)
        datos = df.head(int(limite))
        for i, texto in enumerate(datos[columna].fillna("").astype(str)):
            r = agente.sugerir(texto, n_similares=0)
            filas.append({
                "texto": texto,
                "grupo": r.grupo if r.valido else None,
                "nombre": r.nombre if r.valido else "Sin vocabulario conocido",
                "poco_seguro": r.poco_seguro,
            })
            barra.progress((i + 1) / len(datos))
        resultado = pd.DataFrame(filas)
        st.bar_chart(resultado["nombre"].value_counts(), horizontal=True)
        st.dataframe(resultado, hide_index=True, use_container_width=True)
        st.download_button("Descargar resultado (CSV)", resultado.to_csv(index=False).encode("utf-8"),
                           "triaje_resultado.csv", "text/csv")


def _pestana_grupos(agente: AgenteTriaje) -> None:
    st.subheader("Distribución de las quejas del corpus por grupo")
    dist = agente.distribucion()
    st.bar_chart(dist.set_index("nombre")["quejas"], horizontal=True)
    dist["términos principales"] = [", ".join(agente.terminos[int(g)][:8]) for g in dist["grupo"]]
    st.dataframe(dist, hide_index=True, use_container_width=True)
    st.caption("Los nombres de los grupos son una interpretación humana posterior; no se usaron para entrenar.")


def main() -> None:
    _configurar_pagina()
    st.title("Agente de triaje de quejas en reseñas de productos")
    st.write("Aprendizaje no supervisado (TF-IDF + SVD + K-means) sobre *Women's E-Commerce Clothing Reviews*.")
    try:
        agente = _cargar_agente()
    except FileNotFoundError as error:
        st.error(str(error))
        st.code("python entrenar_agente.py", language="bash")
        st.stop()

    _barra_lateral(agente)
    t1, t2, t3 = st.tabs(["Queja individual", "Lote CSV", "Grupos descubiertos"])
    with t1:
        _pestana_texto(agente)
    with t2:
        _pestana_lote(agente)
    with t3:
        _pestana_grupos(agente)


if __name__ == "__main__":
    main()
