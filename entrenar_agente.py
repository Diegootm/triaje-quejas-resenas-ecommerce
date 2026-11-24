"""Entrena el agente de triaje de punta a punta, sin abrir Jupyter.

Hace lo mismo que los cuadernos 01, 02 y 03 (mismos parámetros y semilla 42),
por lo que produce exactamente el modelo descrito en el informe:

    CSV de Kaggle -> criterio de queja (Rating/Recomendación/VADER) -> 4 675 quejas
    -> preprocesamiento -> TF-IDF (4 675 x 5 000) -> SVD (100) -> K-means (K = 6)

Uso (desde la raíz del repositorio, con el entorno virtual activado):

    python entrenar_agente.py                # entrena con K = 6 (configuración del informe)
    python entrenar_agente.py --barrido      # además imprime la tabla de K = 2..12 (tarda ~2 min)
    python entrenar_agente.py --k 8          # prueba otro número de grupos
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# Evita un aviso de joblib/loky en Windows 11 (no encuentra 'wmic' para contar núcleos).
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))

from codigo import agrupamiento, caracteristicas, etiquetado, evaluacion, preprocesamiento  # noqa: E402
from codigo.triaje import AgenteTriaje  # noqa: E402

RUTA_CSV = RAIZ / "datos" / "crudos" / "Womens Clothing E-Commerce Reviews.csv"
RUTA_ZIP = RAIZ / "datos" / "crudos" / "archive.zip"
DIR_PROCESADOS = RAIZ / "datos" / "procesados"
DIR_MODELOS = RAIZ / "modelos"


def paso(msg: str) -> None:
    print(f"\n=== {msg} ===", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena el agente de triaje de quejas.")
    parser.add_argument("--k", type=int, default=agrupamiento.K_ELEGIDO,
                        help=f"número de grupos, entre {agrupamiento.K_MIN} y {agrupamiento.K_MAX} (por defecto 6)")
    parser.add_argument("--barrido", action="store_true",
                        help="imprime inercia, silueta y la recomendación heurística para K = 2..12")
    args = parser.parse_args()
    try:
        agrupamiento.validar_k(args.k)
    except ValueError as error:
        parser.error(str(error))
    t0 = time.time()

    try:
        preprocesamiento.verificar_recursos_nltk()
    except preprocesamiento.RecursoNltkFaltante as error:
        parser.exit(1, f"error: {error}
")

    paso("1. Carga del dataset y criterio de queja (cuaderno 01)")
    origen = RUTA_CSV if RUTA_CSV.exists() else RUTA_ZIP
    df = preprocesamiento.cargar_resenas(origen)
    print(f"Reseñas con texto: {len(df)}  (origen: {origen.name})")
    df = preprocesamiento.marcar_quejas(df)
    quejas = df[df["es_queja"]].reset_index(drop=True)
    solo_vader = (df["sent_negativo"] & ~df["calif_baja"] & ~df["no_recomienda"]).sum()
    print(f"Quejas seleccionadas: {len(quejas)} de {len(df)} ({len(quejas)/len(df):.1%}); solo por VADER: {solo_vader}")
    DIR_PROCESADOS.mkdir(parents=True, exist_ok=True)
    quejas.to_csv(DIR_PROCESADOS / "resenas_filtradas.csv", index=False, encoding="utf-8", lineterminator="\n")

    paso("2. Preprocesamiento del texto (cuaderno 02)")
    limpias = preprocesamiento.preprocesar_dataframe(quejas)
    print("Ejemplo:", preprocesamiento.preprocesar("This dress did not fit me at all, the fabric was cheap!"))
    columnas = ["Clothing ID", "Age", "Rating", "Recommended IND", "Division Name", "Department Name",
                "Class Name", "vader_compound", "es_queja", "Review Text", "texto", "texto_limpio"]
    limpias[columnas].to_csv(DIR_PROCESADOS / "resenas_limpias.csv", index=False, encoding="utf-8", lineterminator="\n")
    print(f"Reseñas limpias: {len(limpias)}")

    paso("3. TF-IDF + SVD truncada (cuaderno 03)")
    X, vectorizador = caracteristicas.construir_tfidf(limpias["texto_limpio"].tolist())
    X_svd, svd = agrupamiento.reducir_svd(X)
    print(f"Matriz TF-IDF: {X.shape} | densidad {X.nnz / (X.shape[0] * X.shape[1]):.3%}")
    print(f"Matriz SVD: {X_svd.shape} | varianza explicada {svd.explained_variance_ratio_.sum():.1%}")

    if args.barrido:
        paso("3b. Selección de K (codo + silueta)")
        tabla = agrupamiento.seleccionar_k(X_svd)
        print(tabla.to_string(index=False))
        rec = agrupamiento.recomendar_k(tabla)
        if rec.k is None:
            print("Recomendación heurística: ningún K cumple los filtros.")
        else:
            print(f"Recomendación heurística (no es garantía de calidad): K = {rec.k} "
                  f"(mejor silueta entre candidatos: K = {rec.k_mejor_silueta}, {rec.silueta_mejor:.4f})")

    paso(f"4. K-means con K = {args.k}")
    kmeans = agrupamiento.entrenar_kmeans(X_svd, k=args.k)
    etiquetas = kmeans.labels_
    print(f"Iteraciones: {kmeans.n_iter_} | inercia: {kmeans.inertia_:.2f}")
    grupos, tamanos = np.unique(etiquetas, return_counts=True)
    print("Tamaño de cada grupo:", dict(zip(grupos.tolist(), tamanos.tolist())))

    paso("5. Evaluación (cuaderno 04)")
    print(f"Silueta: {evaluacion.coeficiente_silueta(X_svd, etiquetas):.4f}")
    ari = evaluacion.evaluar_estabilidad(X_svd, args.k)["ari"]
    print(f"ARI medio (45 pares, n_init=1): {ari.mean():.4f} ± {ari.std(ddof=0):.4f}")

    paso("6. Guardado de modelos")
    caracteristicas.guardar_vectorizador(vectorizador, DIR_MODELOS / "vectorizador.joblib")
    agrupamiento.guardar_modelo(svd, DIR_MODELOS / "svd_lsa.joblib")
    agrupamiento.guardar_modelo(kmeans, DIR_MODELOS / "modelo_kmeans.joblib")
    print(f"Modelos guardados en {DIR_MODELOS}")

    paso("7. Nombres de los grupos y huella de la partición")
    # Se carga el agente igual que la interfaz para que la huella se calcule
    # sobre la misma partición que verá AgenteTriaje (corpus leído del CSV).
    # Si ya existía un nombres_grupos.json de ESTA misma partición (misma huella),
    # sus nombres se conservan; si no, se usan los manuales (solo con la huella de
    # referencia) o los derivados de los términos.
    agente = AgenteTriaje(DIR_MODELOS)
    nombres = agente.nombres
    for c in sorted(nombres):
        print(f"  Grupo {c}: {nombres[c]}\n           términos: {', '.join(agente.terminos.get(c, [])[:8])}")
    print(f"Huella de la partición: {agente.huella}")
    if agente.origen_nombres == "archivo":
        print(f"Se conservan los nombres del {etiquetado.ARCHIVO_NOMBRES} anterior (misma partición).")
    elif agente.origen_nombres == "referencia":
        print("Coincide con HUELLA_REFERENCIA: se guardan los nombres manuales del informe.")
    elif etiquetado.HUELLA_REFERENCIA is None:
        print("HUELLA_REFERENCIA no está registrada en codigo/etiquetado.py: se guardan nombres derivados de los términos.")
    else:
        print("No coincide con HUELLA_REFERENCIA: se guardan nombres derivados de los términos.")
    if agente.aviso_nombres:
        print("Aviso:", agente.aviso_nombres)
    etiquetado.guardar_nombres(nombres, agente.etiquetas_corpus, DIR_MODELOS)
    print(f"Nombres guardados en {DIR_MODELOS / etiquetado.ARCHIVO_NOMBRES} "
          f"(huella en {etiquetado.ARCHIVO_HUELLA_NOMBRES})")
    print(f"\nListo en {time.time() - t0:.0f} s. Ahora puedes ejecutar: streamlit run app/app_streamlit.py")


if __name__ == "__main__":
    main()
