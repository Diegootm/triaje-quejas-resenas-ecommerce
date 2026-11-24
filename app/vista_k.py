"""Pestaña "Selección de K": la decisión del agente, el barrido K = 2..12 y la comparación con el modelo guardado."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from matplotlib.figure import Figure

from app import estado
from codigo import agrupamiento, config, evaluacion
from codigo.decision_k import DecisionK, explicar_decision
from codigo.presentacion import numero_grupo
from codigo.triaje import AgenteTriaje


def grafico_barrido(tabla: pd.DataFrame, k_actual: int, k_recomendado: int | None) -> Figure:
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


def _decision_del_agente(decision: DecisionK) -> None:
    st.markdown("#### Decisión del agente")
    st.info(explicar_decision(decision))
    st.caption(f"Calculada al entrenar ({decision.fecha}) y guardada en modelos/{config.ARCHIVO_DECISION_K}.")


def _seccion_barrido(base: AgenteTriaje, k_actual: int) -> None:
    st.markdown(f"#### Recalcular el barrido K = {config.K_MIN} … {config.K_MAX}")
    st.caption(
        "Se entrena K-means para cada K sobre la misma representación TF-IDF + SVD del corpus. "
        "**Inercia**: suma de distancias al centroide (siempre baja al subir K; se busca un 'codo'). "
        "**Silueta**: qué tan separado está cada punto de los otros grupos (−1 a 1). "
        "En texto disperso la silueta suele ser baja y casi plana: sirve para comparar K, no como nota de calidad."
    )
    memoria = estado.almacen()
    tabla = memoria.barrido
    if tabla is None:
        if st.button(f"Calcular inercia y silueta para K = {config.K_MIN}…{config.K_MAX}", type="primary"):
            rango = list(config.RANGO_K)
            barra = st.progress(0.0, text=f"Entrenando K = {rango[0]}...")

            def avance(i: int, total: int, fila: dict) -> None:
                siguiente = f" Entrenando K = {rango[i]}..." if i < total else ""
                barra.progress(i / total, text=f"K = {fila['K']} terminado ({i}/{total}).{siguiente}")

            tabla = memoria.calcular_barrido(base, avance)
            barra.empty()
        else:
            st.info("El barrido entrena 11 modelos y puede tardar unos segundos. Los modelos quedan en memoria: "
                    "luego cambiar de K no vuelve a entrenarlos.")
            return

    rec = agrupamiento.recomendar_k(tabla)
    st.pyplot(grafico_barrido(tabla, k_actual, rec.k))

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
            f"**Recomendación de la regla: K = {rec.k}.** Mejor silueta entre los candidatos: "
            f"K = {rec.k_mejor_silueta} ({rec.silueta_mejor:.4f}). Se recomienda el K más pequeño con silueta "
            f"≥ {rec.umbral_silueta:.4f} (hasta {config.TOLERANCIA_SILUETA:.0%} por debajo de la mejor)."
        )
        if rec.k != k_actual:
            st.button(f"Usar K = {rec.k}", on_click=estado.usar_k, args=(rec.k,))
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
    col1.metric("K activo", agente.n_grupos)
    col2.metric("Inercia del modelo", f"{agente.kmeans.inertia_:.2f}")
    ari = evaluacion.indice_rand_ajustado(base.etiquetas_referencia, agente.etiquetas_corpus)
    col3.metric(f"ARI frente a K = {base.k_referencia}", f"{ari:.3f}",
                help="1 = misma partición; cerca de 0 = coincidencia esperable por azar.")
    if agente.n_grupos == base.k_referencia:
        st.caption("Estás viendo el modelo guardado en modelos/.")
        return
    df = pd.DataFrame({
        f"grupo (K = {agente.n_grupos})": [f"{numero_grupo(g):02d}: {agente.nombres[g][:40]}"
                                           for g in agente.etiquetas_corpus],
        f"grupo referencia (K = {base.k_referencia})": [f"{numero_grupo(g):02d}: {base.nombres[g][:30]}"
                                                        for g in base.etiquetas_referencia],
    })
    tabla = evaluacion.comparar_etiquetas_manuales(df, df.columns[1], df.columns[0])
    st.caption("Cada fila es un grupo del K activo; las columnas muestran de qué grupos del modelo "
               "guardado provienen sus quejas (cada fila suma 100 %).")
    st.dataframe(tabla.style.format("{:.0%}").background_gradient(cmap="Blues", axis=None),
                 use_container_width=True)


def pestana_k(base: AgenteTriaje, agente: AgenteTriaje, decision: DecisionK) -> None:
    st.subheader("Selección del número de grupos (K)")
    st.write(
        f"El agente eligió **K = {decision.k_elegido}**; el K activo ahora es **{agente.n_grupos}**. "
        "La distribución y los términos de cada grupo con este K están en la pestaña *Grupos descubiertos*."
    )
    _decision_del_agente(decision)
    st.divider()
    _seccion_barrido(base, agente.n_grupos)
    st.divider()
    _seccion_comparacion(base, agente)
