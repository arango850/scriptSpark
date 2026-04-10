"""
spark-kmeans.py
---------------
Segmentación de Usuarios con K-Means usando Apache Spark
Soporta: MovieLens 100K y MovieLens 1M (detección automática por ruta)

Uso local (pruebas):
    python spark-kmeans.py                             # ML-100K local
    python spark-kmeans.py path/to/ml-1m               # ML-1M local

Uso en cluster (spark-submit) — en Linux usa GCS por defecto (ML-100K):
    spark-submit --master spark://master.us-central1-c.c.lab5-20261.internal:7077 spark-kmeans.py
    spark-submit --master spark://master.us-central1-c.c.lab5-20261.internal:7077 spark-kmeans.py gs://ml-100k/ml-1m
    spark-submit --master spark://master.us-central1-c.c.lab5-20261.internal:7077 spark-kmeans.py gs://otro-bucket/ml-1m
"""

import sys
import os

# En Windows, PySpark lanza el gateway Java a través de spark-class2.cmd, que
# captura la ruta de SPARK_HOME mediante un loop `for /f` de CMD. Ese loop
# corrompe rutas con caracteres no-ASCII (como la é de "Andrés"), dejando el
# classpath incorrecto y causando ClassNotFoundException para SparkSubmit.
#
# Solución: fijar SPARK_HOME al formato corto 8.3 de Windows (sólo ASCII) del
# directorio pyspark instalado via pip. Si no se puede obtener el path corto,
# se usa la ruta larga (que funcionará en sistemas sin ese problema).
#
# En cluster, spark-submit define PYSPARK_GATEWAY_PORT y PySpark omite este
# gateway local por completo, por lo que este bloque es inofensivo allí.
if sys.platform == "win32":
    import importlib.util as _ilu
    _spec = _ilu.find_spec("pyspark")
    if _spec and _spec.origin:
        _pdir = os.path.dirname(os.path.abspath(_spec.origin))
        try:
            import ctypes as _ct
            _buf = _ct.create_unicode_buffer(32768)
            _ct.windll.kernel32.GetShortPathNameW(_pdir, _buf, 32768)
            if _buf.value:
                _pdir = _buf.value
        except Exception:
            pass
        os.environ["SPARK_HOME"] = _pdir
        del _pdir, _spec, _ilu

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, IntegerType, StringType, DoubleType
)
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# ---------------------------------------------------------------------------
# GCS — ruta por defecto del bucket en Google Cloud y conector Hadoop
# ---------------------------------------------------------------------------
GCS_DATASET_PATH    = "gs://ml-100k/ml-100k"   # MovieLens 100K (defecto en Linux)
GCS_DATASET_PATH_1M = "gs://ml-100k/ml-1m"     # MovieLens 1M en el mismo bucket
GCS_CONNECTOR_JAR   = "gcs-connector-hadoop3-latest.jar"
GCS_CONNECTOR_URL = (
    "https://storage.googleapis.com/hadoop-lib/gcs/"
    + GCS_CONNECTOR_JAR
)


def _ensure_gcs_connector() -> str:
    """Descarga el conector GCS a SPARK_HOME/jars/ si no está presente.

    Retorna la ruta absoluta al JAR para que Spark lo distribuya a los
    workers via spark.jars. Si no se puede descargar retorna cadena vacía.
    """
    spark_home = os.environ.get("SPARK_HOME", "")
    if not spark_home:
        print("  ADVERTENCIA: SPARK_HOME no definido; no se puede instalar el conector GCS.")
        return ""
    jar_dest = os.path.join(spark_home, "jars", GCS_CONNECTOR_JAR)
    if not os.path.exists(jar_dest):
        print(f"  Descargando conector GCS → {jar_dest} ...")
        try:
            import urllib.request
            urllib.request.urlretrieve(GCS_CONNECTOR_URL, jar_dest)
            print("  Conector GCS descargado correctamente.")
        except Exception as exc:
            print(f"  ADVERTENCIA: No se pudo descargar el conector GCS: {exc}")
            return ""
    return jar_dest


# ---------------------------------------------------------------------------
# Nombres de géneros (columnas internas, basadas en ML-100K u.item)
# ---------------------------------------------------------------------------
GENRE_NAMES = [
    "unknown", "Action", "Adventure", "Animation", "Childrens",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "FilmNoir", "Horror", "Musical", "Mystery", "Romance",
    "SciFi", "Thriller", "War", "Western",
]

# Mapeo de nombres de género en ML-1M → nombre interno de columna.
# ML-1M usa "Children's", "Film-Noir" y "Sci-Fi"; no tiene "unknown".
ML1M_GENRE_MAP = {
    "Action":      "Action",
    "Adventure":   "Adventure",
    "Animation":   "Animation",
    "Children's":  "Childrens",
    "Comedy":      "Comedy",
    "Crime":       "Crime",
    "Documentary": "Documentary",
    "Drama":       "Drama",
    "Fantasy":     "Fantasy",
    "Film-Noir":   "FilmNoir",
    "Horror":      "Horror",
    "Musical":     "Musical",
    "Mystery":     "Mystery",
    "Romance":     "Romance",
    "Sci-Fi":      "SciFi",
    "Thriller":    "Thriller",
    "War":         "War",
    "Western":     "Western",
}

# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def get_path(base: str, filename: str) -> str:
    """Une un directorio base con un nombre de archivo.
    Funciona tanto con rutas locales como con URIs de GCS (gs://).
    """
    if base.startswith("gs://"):
        return f"{base.rstrip('/')}/{filename}"
    return os.path.join(base, filename)


def detect_format(data_path: str) -> str:
    """Detecta el formato del dataset según el nombre del último segmento de ruta.

    Retorna '1m' si el directorio se llama 'ml-1m' o 'ml_1m' (insensible a
    mayúsculas), '100k' en cualquier otro caso. Funciona con rutas locales
    y URIs de GCS.
    """
    base = data_path.rstrip("/").replace("\\", "/").split("/")[-1].lower()
    if base in ("ml-1m", "ml_1m", "1m"):
        return "1m"
    return "100k"


def create_spark_session(local_mode: bool, gcs_jar: str = "") -> SparkSession:
    """Crea la SparkSession.

    En modo local (pruebas) se fuerza master=local[*].
    En modo cluster el master ya viene configurado por spark-submit,
    por lo que NO se sobreescribe aquí.
    Si gcs_jar es la ruta a un JAR del conector GCS, Spark lo distribuye
    automáticamente a todos los workers y configura el driver de Hadoop.
    """
    builder = SparkSession.builder.appName("UserSegmentation_KMeans")
    if local_mode:
        builder = builder.master("local[*]")
    if gcs_jar:
        builder = (
            builder
            # Distribuir el JAR a todos los executors/workers
            .config("spark.jars", gcs_jar)
            .config("spark.hadoop.fs.gs.impl",
                    "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem")
            .config("spark.hadoop.fs.AbstractFileSystem.gs.impl",
                    "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS")
            .config("spark.hadoop.google.cloud.auth.service.account.enable", "true")
        )
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


# ---------------------------------------------------------------------------
# Carga de datos — soporta ML-100K y ML-1M
# ---------------------------------------------------------------------------

def _read_coloncolon(spark: SparkSession, path: str, col_names: list):
    """Lee un archivo con separador '::' (usado en ML-1M).

    PySpark no garantiza soporte de separadores multi-carácter en CSV,
    así que se lee como texto plano y se parte con split().
    """
    raw = spark.read.text(path)
    split_col = F.split(F.col("value"), "::")
    return raw.select(
        *[split_col[i].alias(name) for i, name in enumerate(col_names)]
    )


def load_ratings(spark: SparkSession, data_path: str, dataset_format: str = "100k"):
    """Carga los ratings.

    ML-100K: u1.base     → userId \\t movieId \\t rating \\t timestamp
    ML-1M:   ratings.dat → userId::movieId::rating::timestamp
    """
    col_names = ["userId", "movieId", "rating", "timestamp"]
    cast_map  = [IntegerType(), IntegerType(), DoubleType(), IntegerType()]

    if dataset_format == "1m":
        df = _read_coloncolon(spark, get_path(data_path, "ratings.dat"), col_names)
    else:
        schema = StructType([StructField(c, t, True) for c, t in zip(col_names, cast_map)])
        df = spark.read.csv(get_path(data_path, "u1.base"), sep="\t", schema=schema)

    for col_name, col_type in zip(col_names, cast_map):
        df = df.withColumn(col_name, F.col(col_name).cast(col_type))
    return df


def load_users(spark: SparkSession, data_path: str, dataset_format: str = "100k"):
    """Carga la información demográfica de los usuarios.

    ML-100K: u.user    → userId | age | gender | occupation | zipCode  (sep='|')
    ML-1M:   users.dat → userId::gender::age::occupation::zipCode
             Nota: en ML-1M el orden de gender y age está invertido.
             La edad es un código categórico (1,18,25,35,45,50,56).
             La ocupación ya es un entero 0-20.
    """
    if dataset_format == "1m":
        col_names = ["userId", "gender", "age", "occupation", "zipCode"]
        df = _read_coloncolon(spark, get_path(data_path, "users.dat"), col_names)
        df = (df
              .withColumn("userId",     F.col("userId").cast(IntegerType()))
              .withColumn("age",        F.col("age").cast(IntegerType()))
              .withColumn("occupation", F.col("occupation").cast(IntegerType()))
              )
    else:
        schema = StructType([
            StructField("userId",     IntegerType(), True),
            StructField("age",        IntegerType(), True),
            StructField("gender",     StringType(),  True),
            StructField("occupation", StringType(),  True),
            StructField("zipCode",    StringType(),  True),
        ])
        df = spark.read.csv(get_path(data_path, "u.user"), sep="|", schema=schema)
    return df


def load_movies(spark: SparkSession, data_path: str, dataset_format: str = "100k"):
    """Carga la información de películas y crea columnas de género binarias.

    ML-100K: u.item      → columnas binarias de género ya vienen en el archivo.
    ML-1M:   movies.dat  → géneros como texto pipe-separated (ej. 'Action|Comedy').
                           Se expanden a las mismas columnas binarias que ML-100K.
    El resultado siempre expone las columnas de GENRE_NAMES.
    """
    if dataset_format == "1m":
        col_names = ["movieId", "title", "genres_str"]
        df = _read_coloncolon(spark, get_path(data_path, "movies.dat"), col_names)
        df = df.withColumn("movieId", F.col("movieId").cast(IntegerType()))
        for ml1m_name, col_name in ML1M_GENRE_MAP.items():
            df = df.withColumn(
                col_name,
                F.when(F.col("genres_str").contains(ml1m_name), 1).otherwise(0)
            )
        df = df.withColumn("unknown", F.lit(0))  # ML-1M no tiene género "unknown"
        return df.drop("genres_str")
    else:
        fields = [
            StructField("movieId",          IntegerType(), True),
            StructField("title",            StringType(),  True),
            StructField("releaseDate",      StringType(),  True),
            StructField("videoReleaseDate", StringType(),  True),
            StructField("imdbUrl",          StringType(),  True),
        ]
        for g in GENRE_NAMES:
            fields.append(StructField(g, IntegerType(), True))
        return spark.read.csv(
            get_path(data_path, "u.item"),
            sep="|", schema=StructType(fields), encoding="ISO-8859-1"
        )


# ---------------------------------------------------------------------------
# 1. Análisis exploratorio
# ---------------------------------------------------------------------------

def exploratory_analysis(ratings, users, movies):
    """Realiza un análisis exploratorio básico del dataset."""
    print("\n" + "=" * 60)
    print("  1. ANÁLISIS EXPLORATORIO DE DATOS")
    print("=" * 60)

    num_users   = ratings.select("userId").distinct().count()
    num_movies  = ratings.select("movieId").distinct().count()
    total_rat   = ratings.count()

    print(f"  Número de usuarios únicos : {num_users}")
    print(f"  Número de películas únicas: {num_movies}")
    print(f"  Total de ratings          : {total_rat}")

    print("\n  Distribución de ratings (1-5):")
    ratings.groupBy("rating").count().orderBy("rating").show(truncate=False)

    print("  Estadísticas descriptivas de ratings:")
    ratings.select("rating").describe().show(truncate=False)

    print("  Distribución de géneros de usuarios:")
    users.groupBy("gender").count().orderBy("gender").show(truncate=False)

    print("  Top 10 ocupaciones más frecuentes:")
    users.groupBy("occupation").count() \
         .orderBy(F.desc("count")).show(10, truncate=False)


# ---------------------------------------------------------------------------
# 2. Construcción de features
# ---------------------------------------------------------------------------

def build_user_features(ratings, users, movies, dataset_format: str = "100k"):
    """Construye el vector de características por usuario.

    Estrategia de feature engineering:
    ----------------------------------
    a) Estadísticas de rating por usuario:
       - Número de películas calificadas (actividad del usuario)
       - Promedio de rating (exigencia del usuario)
       - Desviación estándar del rating (consistencia)
       - Rating mínimo y máximo

    b) Preferencias de género:
       - Para cada uno de los 19 géneros, promedio del rating que el
         usuario asignó a películas de ese género.
       - Si el usuario nunca calificó películas de un género → 0.

    c) Información demográfica:
       - Edad (numérica)
       - Género (M=0, F=1)
       - Ocupación codificada con StringIndexer
    """
    print("\n" + "=" * 60)
    print("  2. CONSTRUCCIÓN DE FEATURES")
    print("=" * 60)

    # --- a) Estadísticas de rating ---
    user_stats = ratings.groupBy("userId").agg(
        F.count("rating").cast(DoubleType()).alias("num_ratings"),
        F.avg("rating").alias("avg_rating"),
        F.stddev("rating").alias("std_rating"),
        F.min("rating").cast(DoubleType()).alias("min_rating"),
        F.max("rating").cast(DoubleType()).alias("max_rating"),
    ).fillna(0.0, subset=["std_rating"])

    # --- b) Preferencias de género ---
    # Unimos ratings con las columnas de género de las películas
    genre_cols = movies.select(["movieId"] + GENRE_NAMES)
    ratings_with_genres = ratings.join(genre_cols, on="movieId", how="left")

    # Para cada género calculamos el promedio del rating en películas de ese género
    genre_agg_exprs = [
        F.avg(F.when(F.col(g) == 1, F.col("rating"))).alias(f"pref_{g}")
        for g in GENRE_NAMES
    ]
    user_genre_prefs = ratings_with_genres.groupBy("userId").agg(*genre_agg_exprs)
    pref_cols = [f"pref_{g}" for g in GENRE_NAMES]
    user_genre_prefs = user_genre_prefs.fillna(0.0, subset=pref_cols)

    # --- c) Información demográfica ---
    users_enc = users.withColumn(
        "gender_num", F.when(F.col("gender") == "F", 1.0).otherwise(0.0)
    )

    # Codificar ocupación numéricamente.
    # ML-100K: texto libre → StringIndexer asigna índices.
    # ML-1M: ya es entero 0-20 → cast directo a double.
    if dataset_format == "1m":
        user_demo = users_enc.select(
            "userId",
            F.col("age").cast(DoubleType()).alias("age"),
            "gender_num",
            F.col("occupation").cast(DoubleType()).alias("occupation_idx"),
        )
    else:
        occ_indexer = StringIndexer(
            inputCol="occupation", outputCol="occupation_idx", handleInvalid="keep"
        )
        users_indexed = occ_indexer.fit(users_enc).transform(users_enc)
        user_demo = users_indexed.select(
            "userId",
            F.col("age").cast(DoubleType()).alias("age"),
            "gender_num",
            "occupation_idx",
        )

    # --- Unión de todas las features ---
    user_features = (
        user_stats
        .join(user_genre_prefs, on="userId", how="inner")
        .join(user_demo,        on="userId", how="inner")
    )

    feature_cols = (
        ["num_ratings", "avg_rating", "std_rating", "min_rating", "max_rating",
         "age", "gender_num", "occupation_idx"]
        + pref_cols
    )

    print(f"\n  Total de features: {len(feature_cols)}")
    for fc in feature_cols:
        print(f"    - {fc}")

    print("\n  Muestra de features (5 usuarios):")
    user_features.select(["userId"] + feature_cols[:6]).show(5, truncate=False)

    return user_features, feature_cols


# ---------------------------------------------------------------------------
# 3. Preparación de datos: ensamble y escalado
# ---------------------------------------------------------------------------

def prepare_features(user_features, feature_cols):
    """Ensambla las columnas en un vector y aplica StandardScaler."""
    assembler = VectorAssembler(
        inputCols=feature_cols, outputCol="features_raw"
    )
    assembled = assembler.transform(user_features)

    scaler = StandardScaler(
        inputCol="features_raw", outputCol="features",
        withMean=True, withStd=True
    )
    scaler_model = scaler.fit(assembled)
    scaled = scaler_model.transform(assembled)
    return scaled


# ---------------------------------------------------------------------------
# 4. Aplicación de K-Means con distintos valores de K
# ---------------------------------------------------------------------------

def apply_kmeans(scaled, k_values=(3, 5, 8)):
    """Entrena K-Means para cada valor de K y evalúa con Silhouette Score.

    El Silhouette Score varía entre -1 (mala agrupación) y 1 (perfecta).
    El WSSSE (Within-Set Sum of Squared Errors) se usa para el método del
    codo: buscamos el punto donde la curva se aplana.
    """
    print("\n" + "=" * 60)
    print("  4. APLICACIÓN DE K-MEANS")
    print("=" * 60)

    evaluator = ClusteringEvaluator(
        featuresCol="features",
        predictionCol="cluster",
        metricName="silhouette",
        distanceMeasure="squaredEuclidean",
    )

    results = {}
    best_k          = None
    best_silhouette = -2.0
    best_model      = None
    best_predictions = None

    for k in k_values:
        print(f"\n  Entrenando K-Means con K={k}...")
        kmeans = KMeans(
            featuresCol="features",
            predictionCol="cluster",
            k=k,
            seed=42,
            maxIter=20,
        )
        model       = kmeans.fit(scaled)
        predictions = model.transform(scaled)

        silhouette = evaluator.evaluate(predictions)
        wssse      = model.summary.trainingCost

        print(f"    K={k} → Silhouette={silhouette:.4f}  |  WSSSE={wssse:,.2f}")

        results[k] = {
            "silhouette":   silhouette,
            "wssse":        wssse,
            "model":        model,
            "predictions":  predictions,
        }

        if silhouette > best_silhouette:
            best_silhouette  = silhouette
            best_k           = k
            best_model       = model
            best_predictions = predictions

    print(f"\n  ➜ Mejor K según Silhouette Score: K={best_k} "
          f"(Silhouette={best_silhouette:.4f})")

    # Resumen comparativo
    print("\n  Resumen comparativo:")
    print(f"  {'K':>4}  {'Silhouette':>12}  {'WSSSE':>16}")
    print("  " + "-" * 36)
    for k in k_values:
        r = results[k]
        marker = " ← mejor" if k == best_k else ""
        print(f"  {k:>4}  {r['silhouette']:>12.4f}  {r['wssse']:>16,.2f}{marker}")

    return results, best_k, best_model, best_predictions


# ---------------------------------------------------------------------------
# 5. Análisis e interpretación de clusters
# ---------------------------------------------------------------------------

def analyze_clusters(best_predictions, users, best_k):
    """Describe cada cluster en términos demográficos y de comportamiento."""
    print("\n" + "=" * 60)
    print(f"  5. ANÁLISIS DE CLUSTERS  (K={best_k})")
    print("=" * 60)

    cluster_col = best_predictions.select("userId", "cluster",
                                          "avg_rating", "num_ratings")
    cluster_users = cluster_col.join(users, on="userId", how="left")

    print("\n  Distribución de usuarios por cluster:")
    cluster_users.groupBy("cluster").count() \
                 .orderBy("cluster").show(truncate=False)

    print("\n  Estadísticas por cluster (edad, % mujeres, ratings):")
    cluster_users.groupBy("cluster").agg(
        F.count("userId").alias("num_users"),
        F.round(F.avg("age"),        1).alias("edad_promedio"),
        F.round(
            F.avg(F.when(F.col("gender") == "F", 1).otherwise(0)) * 100, 1
        ).alias("pct_mujeres"),
        F.round(F.avg("avg_rating"),  3).alias("rating_promedio"),
        F.round(F.avg("num_ratings"), 1).alias("peliculas_calificadas"),
    ).orderBy("cluster").show(truncate=False)

    print("\n  Ocupaciones más frecuentes por cluster:")
    for c in range(best_k):
        print(f"\n    Cluster {c}:")
        cluster_users.filter(F.col("cluster") == c) \
                     .groupBy("occupation").count() \
                     .orderBy(F.desc("count")) \
                     .show(5, truncate=False)

    return cluster_users


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # -----------------------------------------------------------------------
    # Determinar ruta del dataset y modo de ejecución
    # -----------------------------------------------------------------------
    if len(sys.argv) > 1:
        # Ruta explícita pasada como argumento (local o gs://)
        data_path  = sys.argv[1]
        local_mode = False
        print(f"Modo CLUSTER — dataset: {data_path}")
    elif sys.platform != "win32":
        # Linux / GCP: usar el bucket de Cloud Storage por defecto
        data_path  = GCS_DATASET_PATH
        local_mode = False
        print(f"Modo CLUSTER (GCS) — dataset: {data_path}")
    else:
        # Windows: modo local con el dataset incluido en el repositorio
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path  = os.path.join(script_dir, "ml-100k")
        local_mode = True
        print(f"Modo LOCAL — dataset: {data_path}")

    # -----------------------------------------------------------------------
    # Detectar formato del dataset
    # -----------------------------------------------------------------------
    dataset_format = detect_format(data_path)
    print(f"Formato dataset : MovieLens {'1M' if dataset_format == '1m' else '100K'}")

    # -----------------------------------------------------------------------
    # Conector GCS: descarga automática y distribución a workers si es necesario
    # -----------------------------------------------------------------------
    gcs_jar = ""
    if data_path.startswith("gs://"):
        gcs_jar = _ensure_gcs_connector()

    # -----------------------------------------------------------------------
    # Inicializar Spark
    # -----------------------------------------------------------------------
    spark = create_spark_session(local_mode, gcs_jar=gcs_jar)
    print(f"Spark version: {spark.version}")

    # -----------------------------------------------------------------------
    # Carga de datos
    # -----------------------------------------------------------------------
    ratings = load_ratings(spark, data_path, dataset_format)
    users   = load_users(spark, data_path, dataset_format)
    movies  = load_movies(spark, data_path, dataset_format)

    # -----------------------------------------------------------------------
    # 1. Análisis exploratorio
    # -----------------------------------------------------------------------
    exploratory_analysis(ratings, users, movies)

    # -----------------------------------------------------------------------
    # 2 & 3. Feature engineering + preparación
    # -----------------------------------------------------------------------
    user_features, feature_cols = build_user_features(ratings, users, movies, dataset_format)
    scaled = prepare_features(user_features, feature_cols)

    # -----------------------------------------------------------------------
    # 4. K-Means con K = 3, 5, 8
    # -----------------------------------------------------------------------
    results, best_k, best_model, best_predictions = apply_kmeans(
        scaled, k_values=(3, 5, 8)
    )

    # -----------------------------------------------------------------------
    # 5. Análisis de resultados
    # -----------------------------------------------------------------------
    analyze_clusters(best_predictions, users, best_k)

    # Centros de clusters del mejor modelo
    print(f"\n  Centros del mejor modelo (K={best_k}) — primeras 8 dimensiones:")
    labels = ["num_ratings", "avg_rating", "std_rating", "min_rating",
              "max_rating",  "age",         "gender_num", "occupation_idx"]
    for i, center in enumerate(best_model.clusterCenters()):
        vals = "  ".join(f"{v:+.3f}" for v in center[:8])
        print(f"    Cluster {i}: [{vals}]")

    spark.stop()
    print("\n✓ Proceso completado exitosamente.")


if __name__ == "__main__":
    main()
