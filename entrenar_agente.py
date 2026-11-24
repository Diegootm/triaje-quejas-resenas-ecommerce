"""Entrena el agente de triaje de punta a punta, sin abrir Jupyter.

Hace lo mismo que los cuadernos 01, 02 y 03 (mismos parámetros y semilla 42),
por lo que produce exactamente el modelo descrito en el informe:

    CSV de Kaggle -> criterio de queja (Rating/Recomendación/VADER) -> 4 675 quejas
    -> preprocesamiento -> TF-IDF (4 675 x 5 000) -> SVD (100) -> K-means

El agente elige K por sí mismo: entrena K-means para K = 2..12, aplica la regla de
codigo/agrupamiento.recomendar_k (tolerancia en codigo/config.py), guarda la decisión
en modelos/seleccion_k.json y se queda con el modelo ya entrenado del K elegido.

Uso (desde la raíz del repositorio, con el entorno virtual activado):

    python entrenar_agente.py                # el agente elige K (tarda unos minutos)
    python entrenar_agente.py --k 8          # impone K manualmente (queda registrado como "manual")
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from codigo import agrupamiento, caracteristicas, config, decision_k, etiquetado, evaluacion, preprocesamiento  # noqa: E402
from codigo.triaje import AgenteTriaje  # noqa: E402


def paso(msg: str) -> None:
    print(f"\n=== {msg} ===", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena el agente de triaje de quejas.")
    parser.add_argument("--k", type=int, default=None,
                        help=f"impone K manualmente, entre {config.K_MIN} y {config.K_MAX} "
                             "(si se omite, el agente lo elige con el barrido)")
    args = parser.parse_args()
    if args.k is not None:
        try:
            agrupamiento.validar_k(args.k)
        except ValueError as error:
            parser.error(str(error))
    t0 = time.time()

    try:
        preprocesamiento.verificar_recursos_nltk()
    except preprocesamiento.RecursoNltkFaltante as error:
        parser.exit(1, f"error: {error}\n")

    paso("1. Carga del dataset y criterio de queja (cuaderno 01)")
    origen = config.RUTA_CSV if config.RUTA_CSV.exists() else config.RUTA_ZIP
    df = preprocesamiento.cargar_resenas(origen)
    print(f"Reseñas con texto: {len(df)}  (origen: {origen.name})")
    df = preprocesamiento.marcar_quejas(df)
    quejas = df[df["es_queja"]].reset_index(drop=True)
    solo_vader = (df["sent_negativo"] & ~df["calif_baja"] & ~df["no_recomienda"]).sum()
    print(f"Quejas seleccionadas: {len(quejas)} de {len(df)} ({len(quejas)/len(df):.1%}); solo por VADER: {solo_vader}")
    config.DIR_PROCESADOS.mkdir(parents=True, exist_ok=True)
    quejas.to_csv(config.RUTA_FILTRADAS, index=False, encoding="utf-8", lineterminator="\n")

    paso("2. Preprocesamiento del texto (cuaderno 02)")
    limpias = preprocesamiento.preprocesar_dataframe(quejas)
    print("Ejemplo:", preprocesamiento.preprocesar("This dress did not fit me at all, the fabric was cheap!"))
    columnas = ["Clothing ID", "Age", "Rating", "Recommended IND", "Division Name", "Department Name",
                "Class Name", "vader_compound", "es_queja", "Review Text", "texto", "texto_limpio"]
    limpias[columnas].to_csv(config.RUTA_LIMPIAS, index=False, encoding="utf-8", lineterminator="\n")
    print(f"Reseñas limpias: {len(limpias)}")

    paso("3. TF-IDF + SVD truncada (cuaderno 03)")
    X, vectorizador = caracteristicas.construir_tfidf(limpias["texto_limpio"].tolist())
    X_svd, svd = agrupamiento.reducir_svd(X)
    print(f"Matriz TF-IDF: {X.shape} | densidad {X.nnz / (X.shape[0] * X.shape[1]):.3%}")
    print(f"Matriz SVD: {X_svd.shape} | varianza explicada {svd.explained_variance_ratio_.sum():.1%}")

    paso("4. Elección de K por el agente (barrido K = %d..%d)" % (config.K_MIN, config.K_MAX))
    print(f"{'K':>3} {'inercia':>10} {'silueta':>9} {'grupo menor':>12} {'grupo mayor':>12} {'% menor':>8}")

    def mostrar_fila(i: int, total: int, fila: dict) -> None:
        print(f"{fila['K']:>3} {fila['inercia']:>10.2f} {fila['silueta']:>9.4f} {fila['grupo_menor']:>12} "
              f"{fila['grupo_mayor']:>12} {fila['proporcion_menor']:>8.2%}", flush=True)

    t_barrido = time.time()
    tabla, modelos = agrupamiento.barrido_k(X_svd, progreso=mostrar_fila)
    print(f"Barrido terminado en {time.time() - t_barrido:.0f} s")
    decision = decision_k.decidir_k(tabla)
    if args.k is not None:
        decision = decision_k.con_k_manual(decision, args.k)
    print(decision_k.explicar_decision(decision))
    ruta_decision = decision_k.guardar_decision(decision, config.DIR_MODELOS)
    print(f"Decisión guardada en {ruta_decision}")

    k = decision.k_elegido
    paso(f"4b. Modelo K-means con K = {k} (el ya entrenado en el barrido)")
    kmeans = modelos[k]
    etiquetas = kmeans.labels_
    print(f"Iteraciones: {kmeans.n_iter_} | inercia: {kmeans.inertia_:.2f}")
    grupos, tamanos = np.unique(etiquetas, return_counts=True)
    print("Tamaño de cada grupo:", dict(zip(grupos.tolist(), tamanos.tolist())))

    paso("5. Evaluación (cuaderno 04)")
    print(f"Silueta: {evaluacion.coeficiente_silueta(X_svd, etiquetas):.4f}")
    ari = evaluacion.evaluar_estabilidad(X_svd, k)["ari"]
    print(f"ARI medio (45 pares, n_init=1): {ari.mean():.4f} ± {ari.std(ddof=0):.4f}")

    paso("6. Guardado de modelos")
    caracteristicas.guardar_vectorizador(vectorizador, config.DIR_MODELOS / config.ARCHIVO_VECTORIZADOR)
    agrupamiento.guardar_modelo(svd, config.DIR_MODELOS / config.ARCHIVO_SVD)
    agrupamiento.guardar_modelo(kmeans, config.DIR_MODELOS / config.ARCHIVO_KMEANS)
    print(f"Modelos guardados en {config.DIR_MODELOS}")

    paso("7. Nombres de los grupos y huella de la partición")
    # Se carga el agente igual que la interfaz para que la huella se calcule
    # sobre la misma partición que verá AgenteTriaje (corpus leído del CSV).
    # Si ya existía un nombres_grupos.json de ESTA misma partición (misma huella),
    # sus nombres se conservan; si no, se usan los manuales (solo con la huella de
    # referencia) o los derivados de los términos.
    agente = AgenteTriaje(config.DIR_MODELOS)
    nombres = agente.nombres
    for c in sorted(nombres):
        print(f"  Grupo {c}: {nombres[c]}\n           términos: {', '.join(agente.terminos.get(c, [])[:8])}")
    print(f"Huella de la partición: {agente.huella}")
    if agente.origen_nombres == "archivo":
        print(f"Se conservan los nombres del {config.ARCHIVO_NOMBRES} anterior (misma partición).")
    elif agente.origen_nombres == "referencia":
        print("Coincide con HUELLA_REFERENCIA: se guardan los nombres manuales del informe.")
    elif etiquetado.HUELLA_REFERENCIA is None:
        print("HUELLA_REFERENCIA no está registrada en codigo/etiquetado.py: se guardan nombres derivados de los términos.")
    else:
        print("No coincide con HUELLA_REFERENCIA: se guardan nombres derivados de los términos.")
    if agente.aviso_nombres:
        print("Aviso:", agente.aviso_nombres)
    etiquetado.guardar_nombres(nombres, agente.etiquetas_corpus, config.DIR_MODELOS)
    print(f"Nombres guardados en {config.DIR_MODELOS / config.ARCHIVO_NOMBRES} "
          f"(huella en {config.ARCHIVO_HUELLA_NOMBRES})")
    print(f"\nListo en {time.time() - t0:.0f} s. Ahora puedes ejecutar: streamlit run app/app_streamlit.py")


if __name__ == "__main__":
    main()
