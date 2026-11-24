"""Interfaz Streamlit del agente de triaje de quejas.

Carga los modelos entrenados (vectorizador TF-IDF, SVD y K-means) y, para cada
queja nueva, sugiere el grupo más cercano junto con sus términos
representativos y las reseñas más parecidas del corpus.

El número de grupos K se elige en la barra lateral (2 a 12). Al cambiarlo solo
se entrena K-means sobre la representación TF-IDF + SVD ya calculada del corpus;
el modelo de cada K se guarda en memoria y se reutiliza. Clasificar una queja
nueva nunca reentrena el modelo.

Ejecución local (desde la raíz del repositorio):

    streamlit run app/app_streamlit.py
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pandas as pd
import streamlit as st
from matplotlib.figure import Figure
from sklearn.cluster import KMeans

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from codigo import agrupamiento, config, evaluacion  # noqa: E402
from codigo.triaje import AgenteTriaje  # noqa: E402

EJEMPLOS = {
    "Talla": "It runs very small, I had to return it for a larger size.",
    "Calidad y precio": "Poor quality fabric, not worth the price.",
    "Silueta": "The cut is not flattering at all, too many pleats.",
    "Apariencia": "It looked nothing like the picture, very boxy and wide.",
}

CLAVE_K = "k_seleccionado"


def _configurar_pagina() -> None:
    st.set_page_config(page_title="Triaje de quejas de reseñas", page_icon=":material/space_dashboard:", layout="wide")


@st.cache_resource(show_spinner="Cargando modelos entrenados...")
def _cargar_agente() -> AgenteTriaje:
    return AgenteTriaje()


@st.cache_resource(show_spinner=False)
def _almacen() -> dict:
    """Memoria compartida entre sesiones: modelos K-means por K, agentes derivados y barrido."""
    return {"lock": threading.Lock(), "modelos": {}, "agentes": {}, "barrido": None}


# --------------------------------------------------------------------------
# Modelos por K
# --------------------------------------------------------------------------
def _modelo_para_k(agente: AgenteTriaje, k: int) -> KMeans:
    """Modelo de referencia si K coincide; si no, el modelo en memoria o uno nuevo."""
    if k == agente.k_referencia:
        return agente.kmeans_referencia
    almacen = _almacen()
    with almacen["lock"]:
        if k not in almacen["modelos"]:
            with st.spinner(f"Entrenando K-means con K = {k} sobre {len(agente.corpus):,} quejas..."):
                almacen["modelos"][k] = agrupamiento.entrenar_kmeans(agente.X_svd_corpus, k=k)
        return almacen["modelos"][k]


def _agente_para_k(agente: AgenteTriaje, k: int) -> AgenteTriaje:
    if k == agente.k_referencia:
        return agente
    almacen = _almacen()
    if k not in almacen["agentes"]:
        almacen["agentes"][k] = agente.con_kmeans(_modelo_para_k(agente, k))
    return almacen["agentes"][k]


def _ejecutar_barrido(agente: AgenteTriaje) -> None:
    """Entrena K-means para K = 2..12 mostrando el avance real y guarda tabla y modelos."""
    almacen = _almacen()
    rango = list(config.RANGO_K)
    barra = st.progress(0.0, text=f"Entrenando K = {rango[0]}...")

    def avance(i: int, total: int, k: int) -> None:
        siguiente = f" Entrenando K = {rango[i]}..." if i < total else ""
        barra.progress(i / total, text=f"K = {k} terminado ({i}/{total}).{siguiente}")

    with almacen["lock"]:
        tabla, modelos = agrupamiento.barrido_k(agente.X_svd_corpus, rango, progreso=avance)
        for k, modelo in modelos.items():
            almacen["modelos"].setdefault(k, modelo)
        almacen["barrido"] = tabla
    barra.empty()


# --------------------------------------------------------------------------
# Barra lateral
# --------------------------------------------------------------------------
def _selector_k(agente: AgenteTriaje) -> int:
    if CLAVE_K not in st.session_state:
        st.session_state[CLAVE_K] = agente.k_referencia
    with st.sidebar:
        st.header("Número de grupos")
        k = st.slider(
            "K (grupos de K-means)", config.K_MIN, config.K_MAX, key=CLAVE_K,
            help=(f"El modelo guardado usa K = {agente.k_referencia}. Otros valores entrenan solo K-means "
                  "sobre el mismo TF-IDF + SVD; las quejas nuevas se asignan con el K elegido."),
        )
    return agrupamiento.validar_k(k, len(agente.corpus))


def _barra_lateral(agente: AgenteTriaje) -> None:
    with st.sidebar:
        origen = "modelo guardado en modelos/" if agente.n_grupos == agente.k_referencia else "entrenado en memoria"
        nombres = {
            "archivo": "guardados en modelos/nombres_grupos.json (huella verificada)",
            "referencia": "manuales del informe (huella de referencia verificada)",
        }.get(agente.origen_nombres, "derivados de los términos")
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
# Pestañas existentes
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
        resultado.insert(0, "K", agente.n_grupos)
        st.bar_chart(resultado["nombre"].value_counts(), horizontal=True)
        st.dataframe(resultado, hide_index=True, use_container_width=True)
        st.download_button("Descargar resultado (CSV)", resultado.to_csv(index=False).encode("utf-8"),
                           f"triaje_resultado_k{agente.n_grupos}.csv", "text/csv")


def _pestana_grupos(agente: AgenteTriaje) -> None:
    st.subheader(f"Distribución de las quejas del corpus por grupo (K = {agente.n_grupos})")
    dist = agente.distribucion()
    st.bar_chart(dist.set_index("nombre")["quejas"], horizontal=True)
    dist["términos principales"] = [", ".join(agente.terminos[int(g)][:8]) for g in dist["grupo"]]
    st.dataframe(dist, hide_index=True, use_container_width=True)
    if agente.origen_nombres == "referencia":
        st.caption("Los nombres de los grupos son una interpretación humana posterior del modelo de referencia; "
                   "no se usaron para entrenar.")
    elif agente.origen_nombres == "archivo":
        st.caption("Los nombres provienen de modelos/nombres_grupos.json y pertenecen a esta misma partición "
                   "(huella verificada); no se usaron para entrenar.")
    else:
        st.caption("Los nombres se generan con los 3 términos de mayor peso TF-IDF de cada grupo. "
                   "Son orientativos: conviene revisarlos leyendo reseñas de cada grupo.")


# --------------------------------------------------------------------------
# Pestaña nueva: selección de K
# --------------------------------------------------------------------------
def _grafico_barrido(tabla: pd.DataFrame, k_actual: int, k_recomendado: int | None) -> Figure:
    fig = Figure(figsize=(10, 3.4))
    ejes = fig.subplots(1, 2)
    for eje, columna, titulo in ((ejes[0], "inercia", "Inercia (método del codo)"),
                                 (ejes[1], "silueta", f"Silueta (muestra de {config.MUESTRA_SILUETA})")):
        eje.plot(tabla["K"], tabla[columna], "o-", color="#4C78A8")
        eje.axvline(k_actual, color="#E45756", linewidth=2, label=f"K seleccionado = {k_actual}")
        if k_recomendado is not None and k_recomendado != k_actual:
            eje.axvline(k_recomendado, color="#54A24B", linestyle="--", label=f"K recomendado = {k_recomendado}")
        eje.set_title(titulo)
        eje.set_xlabel("K")
        eje.set_xticks(list(tabla["K"]))
        eje.grid(alpha=0.3)
    ejes[1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def _usar_k(k: int) -> None:
    st.session_state[CLAVE_K] = k


def _seccion_barrido(agente: AgenteTriaje, k_actual: int) -> None:
    st.markdown(f"#### Comparar K = {config.K_MIN} … {config.K_MAX}")
    st.caption(
        "Se entrena K-means para cada K sobre la misma representación TF-IDF + SVD del corpus. "
        "**Inercia**: suma de distancias al centroide (siempre baja al subir K; se busca un 'codo'). "
        "**Silueta**: qué tan separado está cada punto de los otros grupos (−1 a 1). "
        "En texto disperso la silueta suele ser baja y casi plana: sirve para comparar K, no como nota de calidad."
    )
    tabla = _almacen()["barrido"]
    if tabla is None:
        if st.button(f"Calcular inercia y silueta para K = {config.K_MIN}…{config.K_MAX}", type="primary"):
            _ejecutar_barrido(agente)
            tabla = _almacen()["barrido"]
        else:
            st.info("El barrido entrena 11 modelos y puede tardar. Los modelos quedan en memoria: "
                    "luego cambiar de K no vuelve a entrenarlos.")
            return

    rec = agrupamiento.recomendar_k(tabla)
    st.pyplot(_grafico_barrido(tabla, k_actual, rec.k))

    vista = tabla.assign(seleccionado=tabla["K"] == k_actual, recomendado=tabla["K"] == rec.k)
    st.dataframe(
        vista, hide_index=True, use_container_width=True,
        column_config={
            "inercia": st.column_config.NumberColumn(format="%.2f"),
            "silueta": st.column_config.NumberColumn(format="%.4f"),
            "proporcion_menor": st.column_config.NumberColumn("proporción grupo menor", format="%.3f"),
        },
    )

    if rec.k is None:
        st.warning("Ningún K cumple los filtros de la heurística; elige K manualmente.")
    else:
        st.info(
            f"**Recomendación heurística: K = {rec.k}.** Mejor silueta entre los candidatos: "
            f"K = {rec.k_mejor_silueta} ({rec.silueta_mejor:.4f}). Se recomienda el K más pequeño con silueta "
            f"≥ {rec.umbral_silueta:.4f} (hasta {config.TOLERANCIA_SILUETA:.0%} por debajo de la mejor)."
        )
        if rec.k != k_actual:
            st.button(f"Usar K = {rec.k}", on_click=_usar_k, args=(rec.k,))
    with st.expander("Cómo se calcula la recomendación y sus limitaciones"):
        st.markdown(
            f"1. Se descartan K < {config.K_MIN_RECOMENDADO} (dos grupos no distinguen tipos de queja) y los K "
            f"cuyo grupo más chico tiene menos del {config.PROPORCION_MINIMA_GRUPO:.0%} del corpus.\n"
            "2. Entre los restantes se busca la mejor silueta.\n"
            f"3. Se elige el K más pequeño cuya silueta esté a menos de {config.TOLERANCIA_SILUETA:.0%} "
            "de la mejor (preferir modelos más simples cuando la diferencia es pequeña).\n\n"
            "**Limitaciones:** es una regla heurística, no una prueba de calidad. La silueta se calcula sobre una "
            "muestra y las diferencias entre K son pequeñas, así que pueden deberse al azar. No mide si los grupos "
            "son interpretables ni estables: la decisión final debe revisar los términos y reseñas de cada grupo."
        )
        if rec.descartados:
            st.markdown("**K descartados:** " + "; ".join(f"K = {k}: {motivo}" for k, motivo in rec.descartados.items()))


def _seccion_comparacion(base: AgenteTriaje, agente: AgenteTriaje) -> None:
    st.markdown(f"#### Cómo cambia el agrupamiento respecto al modelo guardado (K = {base.k_referencia})")
    col1, col2, col3 = st.columns(3)
    col1.metric("K seleccionado", agente.n_grupos)
    col2.metric("Inercia del modelo", f"{agente.kmeans.inertia_:.2f}")
    ari = evaluacion.indice_rand_ajustado(base.etiquetas_referencia, agente.etiquetas_corpus)
    col3.metric(f"ARI frente a K = {base.k_referencia}", f"{ari:.3f}",
                help="1 = misma partición; cerca de 0 = coincidencia esperable por azar.")
    if agente.n_grupos == base.k_referencia:
        st.caption("Estás viendo el modelo guardado en modelos/.")
        return
    df = pd.DataFrame({
        f"grupo (K = {agente.n_grupos})": [f"{g:02d}: {agente.nombres[g][:40]}" for g in agente.etiquetas_corpus],
        f"grupo referencia (K = {base.k_referencia})": [f"{g:02d}: {base.nombres[g][:30]}"
                                                        for g in base.etiquetas_referencia],
    })
    tabla = evaluacion.comparar_etiquetas_manuales(df, df.columns[1], df.columns[0])
    st.caption("Cada fila es un grupo del K seleccionado; las columnas muestran de qué grupos del modelo "
               "guardado provienen sus quejas (cada fila suma 100 %).")
    st.dataframe(tabla.style.format("{:.0%}").background_gradient(cmap="Blues", axis=None),
                 use_container_width=True)


def _pestana_k(base: AgenteTriaje, agente: AgenteTriaje) -> None:
    st.subheader("Selección del número de grupos (K)")
    st.write(
        f"K se elige en la barra lateral. El K actual es **{agente.n_grupos}**. La distribución y los términos de "
        "cada grupo con este K están en la pestaña *Grupos descubiertos*."
    )
    _seccion_barrido(base, agente.n_grupos)
    st.divider()
    _seccion_comparacion(base, agente)


def main() -> None:
    _configurar_pagina()
    st.title("Agente de triaje de quejas en reseñas de productos")
    st.write("Aprendizaje no supervisado (TF-IDF + SVD + K-means) sobre *Women's E-Commerce Clothing Reviews*.")
    try:
        base = _cargar_agente()
    except FileNotFoundError as error:
        st.error(str(error))
        st.code("python entrenar_agente.py", language="bash")
        st.stop()

    k = _selector_k(base)
    agente = _agente_para_k(base, k)
    _barra_lateral(agente)
    t1, t2, t3, t4 = st.tabs(["Queja individual", "Lote CSV", "Grupos descubiertos", "Selección de K"])
    with t1:
        _pestana_texto(agente)
    with t2:
        _pestana_lote(agente)
    with t3:
        _pestana_grupos(agente)
    with t4:
        _pestana_k(base, agente)


if __name__ == "__main__":
    main()
