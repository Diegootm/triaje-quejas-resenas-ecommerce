"""Pestañas de triaje: queja individual, lote CSV y grupos descubiertos, más el resumen de la barra lateral."""

from __future__ import annotations

from typing import Callable

import pandas as pd
import streamlit as st

from codigo.decision_k import DecisionK
from codigo.triaje import AgenteTriaje

EJEMPLOS = {
    "Talla": "It runs very small, I had to return it for a larger size.",
    "Calidad y precio": "Poor quality fabric, not worth the price.",
    "Silueta": "The cut is not flattering at all, too many pleats.",
    "Apariencia": "It looked nothing like the picture, very boxy and wide.",
}


def barra_lateral(agente: AgenteTriaje, decision: DecisionK) -> None:
    with st.sidebar:
        if agente.n_grupos != decision.k_elegido:
            origen = "K manual, entrenado en memoria"
        elif agente.n_grupos == agente.k_referencia:
            origen = "modelo guardado en modelos/"
        else:
            origen = "K del agente, entrenado en memoria"
        nombres = {
            "archivo": "automáticos + guardados en modelos/nombres_grupos.json (huella verificada)",
            "referencia": "automáticos + nombres del informe (huella de referencia verificada)",
        }.get(agente.origen_nombres, "automáticos (términos TF-IDF)")
        st.divider()
        st.markdown(
            f"- K-means: **K = {agente.n_grupos}** ({origen})\n"
            f"- Nombres de grupos: {nombres}\n"
            f"- Vocabulario TF-IDF: **{len(agente.vectorizador.vocabulary_):,}** términos (1-2 gramas)\n"
            f"- SVD truncada: **{agente.svd.n_components}** componentes\n"
            f"- Corpus de quejas: **{len(agente.corpus):,}** reseñas"
        )
        if agente.aviso_nombres:
            st.caption(agente.aviso_nombres)
        st.divider()
        st.markdown("**Criterio de queja**")
        st.caption("Rating ≤ 2  **o**  Recommended IND = 0  **o**  VADER compound ≤ −0.05")
        st.divider()
        st.caption(
            "El agente solo **sugiere** un grupo: no resuelve la queja ni responde al cliente. "
            "El modelo fue entrenado con reseñas en **inglés**."
        )


# --------------------------------------------------------------------------
# Queja individual
# --------------------------------------------------------------------------
def _mostrar_resultado(agente: AgenteTriaje, texto: str) -> None:
    r = agente.sugerir(texto, n_similares=5)
    st.markdown(f"**Texto preprocesado:** `{r.texto_limpio or '(vacío)'}`")
    if not r.valido:
        st.error(r.aviso)
        st.caption("Prueba escribir la queja en inglés, por ejemplo: " + EJEMPLOS["Talla"])
        return

    col1, col2 = st.columns([3, 2])
    with col1:
        st.success(f"**Grupo sugerido (K = {agente.n_grupos}): {r.grupo} — {r.nombre}**")
        if r.nombre_informe:
            st.caption(f"Nombre del informe: {r.nombre_informe}")
        if r.aviso:
            st.warning(r.aviso)
        st.markdown("**Términos representativos del grupo:** " + ", ".join(f"`{t}`" for t in r.terminos[:8]))
    with col2:
        tabla = pd.DataFrame({
            "grupo": [f"{i}: {agente.nombres[i][:28]}" for i in range(agente.n_grupos)],
            "distancia": r.distancias,
        })
        st.markdown("**Distancia a cada centroide** (menor = más parecido)")
        st.bar_chart(tabla, x="grupo", y="distancia", horizontal=True, height=max(240, 24 * agente.n_grupos))

    st.markdown("**Reseñas más parecidas del corpus** (similitud del coseno)")
    st.dataframe(
        pd.DataFrame(r.similares).rename(columns={"similitud": "coseno", "texto": "reseña"}),
        hide_index=True, use_container_width=True,
        column_config={"coseno": st.column_config.NumberColumn(format="%.3f")},
    )


def pestana_texto(agente: AgenteTriaje) -> None:
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


# --------------------------------------------------------------------------
# Lote CSV
# --------------------------------------------------------------------------
def clasificar_lote(
    agente: AgenteTriaje, textos: list[str], progreso: Callable[[float], None] | None = None,
) -> pd.DataFrame:
    """Asigna cada texto a un grupo con el K del agente activo (sin reentrenar nada)."""
    filas = []
    for i, texto in enumerate(textos):
        r = agente.sugerir(texto, n_similares=0)
        filas.append({
            "texto": texto,
            "grupo": r.grupo if r.valido else None,
            "nombre": r.nombre if r.valido else "Sin vocabulario conocido",
            "poco_seguro": r.poco_seguro,
        })
        if progreso is not None:
            progreso((i + 1) / len(textos))
    resultado = pd.DataFrame(filas)
    resultado.insert(0, "K", agente.n_grupos)
    return resultado


def pestana_lote(agente: AgenteTriaje) -> None:
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
        barra = st.progress(0.0)
        textos = df.head(int(limite))[columna].fillna("").astype(str).tolist()
        resultado = clasificar_lote(agente, textos, progreso=barra.progress)
        st.bar_chart(resultado["nombre"].value_counts(), horizontal=True)
        st.dataframe(resultado, hide_index=True, use_container_width=True)
        st.download_button("Descargar resultado (CSV)", resultado.to_csv(index=False).encode("utf-8"),
                           f"triaje_resultado_k{agente.n_grupos}.csv", "text/csv")


# --------------------------------------------------------------------------
# Grupos descubiertos
# --------------------------------------------------------------------------
def pestana_grupos(agente: AgenteTriaje) -> None:
    st.subheader(f"Distribución de las quejas del corpus por grupo (K = {agente.n_grupos})")
    dist = agente.distribucion()
    st.bar_chart(dist.set_index("nombre")["quejas"], horizontal=True)
    dist["términos principales"] = [", ".join(agente.terminos[int(g)][:8]) for g in dist["grupo"]]
    st.dataframe(dist, hide_index=True, use_container_width=True)
    if agente.origen_nombres == "referencia":
        st.caption("Nombre principal: automático (términos TF-IDF). \"Nombre del informe\": interpretación humana "
                   "posterior del modelo de referencia; ninguno se usó para entrenar.")
    elif agente.origen_nombres == "archivo":
        st.caption("Nombre principal: automático (términos TF-IDF). \"Nombre del informe\" viene de "
                   "modelos/nombres_grupos.json y pertenece a esta misma partición (huella verificada).")
    else:
        st.caption("Los nombres se generan con los 3 términos de mayor peso TF-IDF de cada grupo. "
                   "Son orientativos: conviene revisarlos leyendo reseñas de cada grupo.")
