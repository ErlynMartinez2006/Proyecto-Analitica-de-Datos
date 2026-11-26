"""
preparacion.py
==============
Funciones de preparación de datos para el dataset "Resultados únicos Saber 11".

El cuadernillo ``02_Preparacion_Datos.ipynb`` importa este módulo y muestra,
paso a paso, el efecto de cada función sobre el dataset.

Orden del flujo
---------------
1. Limpieza (reglas fijas, no dependen de los datos):
   cargar_datos -> eliminar_duplicados -> filtrar_registros_validos ->
   corregir_puntajes -> crear_variables -> transformar_categoricas ->
   eliminar_columnas_innecesarias
2. Objetivo y variables: filtrar_objetivo_valido -> seleccionar_variables
3. Partición: dividir_train_test
4. Con parámetros calculados solo en train (evita fuga de datos):
   ajustar_imputacion -> fillna, ajustar_escalado -> aplicar_escalado
5. Guardado: guardar_datasets

Uso rápido
----------
Desde un cuadernillo en cuadernillos/ (la raíz del proyecto se agrega al path):

>>> import sys; sys.path.insert(0, '..')
>>> from src.preparacion import *
>>> df = cargar_datos()   # lee data/raw/saber11.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Rutas del proyecto (relativas a este archivo, funcionan desde cualquier cuadernillo)
# ---------------------------------------------------------------------------

RAIZ = Path(__file__).resolve().parents[1]
RUTA_RAW = RAIZ / "data" / "raw" / "saber11.csv"
CARPETA_MODIFICADO = RAIZ / "data" / "modificado"
CARPETA_ENTRENAMIENTO = RAIZ / "data" / "entrenamiento"
CARPETA_TEST = RAIZ / "data" / "test"

# ---------------------------------------------------------------------------
# Constantes del dataset
# ---------------------------------------------------------------------------

OBJETIVO = "PUNT_GLOBAL"

PUNTAJES_AREA = [
    "PUNT_INGLES",
    "PUNT_MATEMATICAS",
    "PUNT_SOCIALES_CIUDADANAS",
    "PUNT_C_NATURALES",
    "PUNT_LECTURA_CRITICA",
]

# Orden ordinal de la educación de los padres
ORDEN_EDUCACION = {
    "Ninguno": 0,
    "Primaria incompleta": 1,
    "Primaria completa": 2,
    "Secundaria (Bachillerato) incompleta": 3,
    "Secundaria (Bachillerato) completa": 4,
    "Técnica o tecnológica incompleta": 5,
    "Técnica o tecnológica completa": 6,
    "Educación profesional incompleta": 7,
    "Educación profesional completa": 8,
    "Postgrado": 9,
}

ESTRATOS = {f"Estrato {i}": i for i in range(1, 7)}  # "Sin Estrato" queda como nulo

CUARTOS = {
    "Uno": 1, "Dos": 2, "Tres": 3, "Cuatro": 4, "Cinco": 5, "Seis": 6,
    "Seis o mas": 6, "Siete": 7, "Ocho": 8, "Nueve": 9, "Diez o más": 10,
}

# A partir de 2017 la encuesta cambió la forma de preguntar el número de
# personas del hogar (de número exacto a rangos). Se unifican en 5 rangos.
PERSONAS_HOGAR = {
    "Una": 1, "Dos": 1, "1 a 2": 1,
    "Tres": 2, "Cuatro": 2, "3 a 4": 2,
    "Cinco": 3, "Seis": 3, "5 a 6": 3,
    "Siete": 4, "Ocho": 4, "7 a 8": 4,
    "Nueve": 5, "Diez": 5, "Once": 5, "Doce o más": 5, "9 o más": 5,
}

SI_NO = {"No": 0, "Si": 1}

# nueva columna: (columna original, mapeo a número)
ORDINALES = {
    "ESTRATO": ("FAMI_ESTRATOVIVIENDA", ESTRATOS),
    "EDUC_MADRE": ("FAMI_EDUCACIONMADRE", ORDEN_EDUCACION),
    "EDUC_PADRE": ("FAMI_EDUCACIONPADRE", ORDEN_EDUCACION),
    "CUARTOS_HOGAR": ("FAMI_CUARTOSHOGAR", CUARTOS),
    "PERSONAS_HOGAR": ("FAMI_PERSONASHOGAR", PERSONAS_HOGAR),
}
BINARIAS = {
    "COLE_NO_OFICIAL": ("COLE_NATURALEZA", {"OFICIAL": 0, "NO OFICIAL": 1}),
    "COLE_RURAL": ("COLE_AREA_UBICACION", {"URBANO": 0, "RURAL": 1}),
    "COLE_BILINGUE": ("COLE_BILINGUE", {"N": 0, "S": 1}),
    "COLE_SEDE_PRINCIPAL": ("COLE_SEDE_PRINCIPAL", {"N": 0, "S": 1}),
    "ESTU_FEMENINO": ("ESTU_GENERO", {"M": 0, "F": 1}),
    "TIENE_AUTOMOVIL": ("FAMI_TIENEAUTOMOVIL", SI_NO),
    "TIENE_COMPUTADOR": ("FAMI_TIENECOMPUTADOR", SI_NO),
    "TIENE_INTERNET": ("FAMI_TIENEINTERNET", SI_NO),
    "TIENE_LAVADORA": ("FAMI_TIENELAVADORA", SI_NO),
}
# prefijo de las columnas one-hot: columna original
NOMINALES = {"CARACTER": "COLE_CARACTER", "JORNADA": "COLE_JORNADA",
             "MCPIO": "COLE_MCPIO_UBICACION"}

# Columnas sin valor para el modelo. Se eliminan después de crear las variables
# derivadas que las necesitan.
COLUMNAS_A_ELIMINAR = [
    # identificadores, nombres y estado del registro
    "ESTU_CONSECUTIVO", "ESTU_TIPODOCUMENTO", "COLE_NOMBRE_ESTABLECIMIENTO",
    "COLE_NOMBRE_SEDE", "COLE_CODIGO_ICFES", "COLE_COD_DANE_ESTABLECIMIENTO",
    "COLE_COD_DANE_SEDE", "ESTU_ESTADOINVESTIGACION",
    # códigos que repiten el nombre del lugar
    "COLE_COD_DEPTO_UBICACION", "COLE_COD_MCPIO_UBICACION",
    "ESTU_COD_DEPTO_PRESENTACION", "ESTU_COD_MCPIO_PRESENTACION",
    "ESTU_COD_RESIDE_DEPTO", "ESTU_COD_RESIDE_MCPIO",
    # residencia (resumida en RESIDE_MISMO_MCPIO)
    "ESTU_DEPTO_RESIDE", "ESTU_MCPIO_RESIDE", "ESTU_PAIS_RESIDE",
    # constantes o casi constantes
    "COLE_CALENDARIO", "COLE_DEPTO_UBICACION", "ESTU_DEPTO_PRESENTACION",
    "ESTU_MCPIO_PRESENTACION", "ESTU_ESTUDIANTE", "ESTU_PRIVADO_LIBERTAD",
    "COLE_GENERO", "ESTU_NACIONALIDAD",
    # resumidas en EDAD y PUNT_INGLES
    "ESTU_FECHANACIMIENTO", "DESEMP_INGLES",
]

COLUMNAS_A_ESCALAR = ["EDAD", "ESTRATO", "EDUC_MADRE", "EDUC_PADRE", "CUARTOS_HOGAR",
                      "PERSONAS_HOGAR", "PUNT_INGLES", "ANIO"]


# ---------------------------------------------------------------------------
# 1. Carga y limpieza
# ---------------------------------------------------------------------------

def cargar_datos(ruta: str | Path = RUTA_RAW) -> pd.DataFrame:
    """Lee el CSV de Saber 11 (por defecto data/raw/saber11.csv; utf-8 con BOM)."""
    return pd.read_csv(ruta, encoding="utf-8-sig", low_memory=False)


def eliminar_duplicados(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina filas idénticas (en el EDA: 5.759, de los periodos 2019-4 y 2022-4)."""
    return df.drop_duplicates().reset_index(drop=True)


def filtrar_registros_validos(df: pd.DataFrame) -> pd.DataFrame:
    """Conserva solo los registros con estado 'PUBLICAR'.

    Los demás estados (lectura tardía, validez en oficina jurídica, identidad
    no comprobada) son registros en investigación y sus puntajes no son oficiales.
    """
    return df[df["ESTU_ESTADOINVESTIGACION"] == "PUBLICAR"].reset_index(drop=True)


def corregir_puntajes(df: pd.DataFrame) -> pd.DataFrame:
    """Corrige valores imposibles en los puntajes.

    * ``PUNT_INGLES == -1``: valor centinela de "sin dato" -> NaN.
    * Puntajes de área fuera de [0, 100] (inglés hasta 117,29 y matemáticas 101)
      -> NaN, porque la escala oficial por prueba es de 0 a 100.
    Los NaN resultantes se imputan más adelante con la mediana de train.
    """
    df = df.copy()
    for columna in PUNTAJES_AREA:
        fuera = (df[columna] < 0) | (df[columna] > 100)
        df.loc[fuera, columna] = np.nan
    return df


def convertir_fecha_nacimiento(serie: pd.Series) -> pd.Series:
    """Convierte 'dd/mm/aaaa' a fecha; lo que no se pueda convertir queda NaT."""
    return pd.to_datetime(serie, format="%d/%m/%Y", errors="coerce")


# ---------------------------------------------------------------------------
# 2. Ingeniería de variables y codificación
# ---------------------------------------------------------------------------

def crear_variables(df: pd.DataFrame, edad_min: int = 12, edad_max: int = 70) -> pd.DataFrame:
    """Crea variables derivadas a partir de columnas que luego se eliminan.

    * ``ANIO``: año del examen, sale de PERIODO (p. ej. 20142 -> 2014).
    * ``EDAD``: años cumplidos al presentar el examen. Si sale fuera de
      [edad_min, edad_max] (fechas como el año 0009) queda NaN.
    * ``RESIDE_MISMO_MCPIO``: 1 si el estudiante vive en el municipio del colegio.
    """
    df = df.copy()
    df["ANIO"] = df["PERIODO"] // 10

    nacimiento = convertir_fecha_nacimiento(df["ESTU_FECHANACIMIENTO"])
    edad = df["ANIO"] - nacimiento.dt.year
    df["EDAD"] = edad.where(edad.between(edad_min, edad_max))

    mismo = df["ESTU_COD_RESIDE_MCPIO"] == df["COLE_COD_MCPIO_UBICACION"]
    df["RESIDE_MISMO_MCPIO"] = mismo.astype(float).where(df["ESTU_COD_RESIDE_MCPIO"].notna())
    return df


def agrupar_categorias_raras(serie: pd.Series, min_frecuencia: int = 300,
                             etiqueta: str = "OTROS") -> pd.Series:
    """Agrupa en ``etiqueta`` las categorías con menos de ``min_frecuencia`` filas."""
    conteo = serie.value_counts()
    raras = conteo[conteo < min_frecuencia].index
    return serie.where(~serie.isin(raras), etiqueta)


def transformar_categoricas(df: pd.DataFrame, min_frecuencia_mcpio: int = 300) -> pd.DataFrame:
    """Convierte las variables de texto a numéricas (ver ORDINALES, BINARIAS, NOMINALES).

    * Ordinales y binarias: se mapean a número; "Sin Estrato" queda como nulo.
    * Nominales: one-hot; los municipios con menos de ``min_frecuencia_mcpio``
      estudiantes se agrupan en 'OTROS' y los nulos pasan a 'SIN_DATO'.
    * ``EDUC_*_NO_SABE``: indicador de las respuestas "No sabe" o "No aplica" en la
      educación de los padres, que al pasar a nulo perderían esa información.
    """
    df = df.copy()
    numericas = {**ORDINALES, **BINARIAS}
    for nueva, (original, mapa) in numericas.items():
        df[nueva] = df[original].map(mapa)
    for padre in ("MADRE", "PADRE"):
        df[f"EDUC_{padre}_NO_SABE"] = (
            df[f"FAMI_EDUCACION{padre}"].isin(["No sabe", "No Aplica"]).astype(int))

    series = {prefijo: df[original] for prefijo, original in NOMINALES.items()}
    series["MCPIO"] = agrupar_categorias_raras(series["MCPIO"], min_frecuencia_mcpio)
    dummies = [pd.get_dummies(serie.fillna("SIN_DATO"), prefix=prefijo, dtype=int)
               for prefijo, serie in series.items()]

    reemplazadas = [original for nueva, (original, _) in numericas.items() if original != nueva]
    return pd.concat([df.drop(columns=reemplazadas + list(NOMINALES.values())), *dummies], axis=1)


def eliminar_columnas_innecesarias(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina identificadores, códigos redundantes y columnas constantes."""
    return df.drop(columns=COLUMNAS_A_ELIMINAR)


def columnas_constantes(df: pd.DataFrame) -> list[str]:
    """Lista las columnas con un único valor (sirve para justificar su eliminación)."""
    return [c for c in df.columns if df[c].nunique(dropna=False) <= 1]


# ---------------------------------------------------------------------------
# 3. Objetivo, variables y partición
# ---------------------------------------------------------------------------

def filtrar_objetivo_valido(df: pd.DataFrame, objetivo: str = OBJETIVO,
                            minimo: float = 1) -> pd.DataFrame:
    """Deja solo las filas con objetivo conocido y válido.

    Los periodos anteriores a 2014-2 no tienen puntaje global (nulo). ``minimo``
    descarta además puntajes globales iguales a cero.
    """
    return df[df[objetivo].notna() & (df[objetivo] >= minimo)].reset_index(drop=True)


def seleccionar_variables(df: pd.DataFrame, objetivo: str = OBJETIVO,
                          incluir_puntajes_area: bool = False
                          ) -> tuple[pd.DataFrame, pd.Series]:
    """Separa X (predictoras) e y (objetivo).

    Por defecto se excluyen los puntajes por área: el puntaje global se calcula
    a partir de ellos, así que incluirlos haría que el modelo "adivine" el
    resultado sin aprender del contexto (fuga de datos). Con
    ``incluir_puntajes_area=True`` se conserva PUNT_INGLES, que no entra en ese cálculo.
    """
    excluir = [c for c in PUNTAJES_AREA if not (incluir_puntajes_area and c == "PUNT_INGLES")]
    # PERIODO ya está resumido en ANIO
    return df.drop(columns=[objetivo, "PERIODO", *excluir]), df[objetivo]


def dividir_train_test(X: pd.DataFrame, y: pd.Series, proporcion_test: float = 0.2,
                       semilla: int = 42):
    """Partición aleatoria en train/test (80/20 por defecto, reproducible)."""
    return train_test_split(X, y, test_size=proporcion_test, random_state=semilla)


# ---------------------------------------------------------------------------
# 4. Imputación y escalado (se ajustan SOLO con train)
# ---------------------------------------------------------------------------

def ajustar_imputacion(X_train: pd.DataFrame) -> dict[str, float]:
    """Valor de relleno de cada columna con nulos, calculado con train.

    Binarias -> moda; numéricas y ordinales -> mediana. Se aplica con
    ``X.fillna(valores)``.
    """
    valores = {}
    for columna in X_train.columns[X_train.isna().any()]:
        serie = X_train[columna].dropna()
        valores[columna] = float(serie.mode().iloc[0] if serie.nunique() <= 2 else serie.median())
    return valores


def ajustar_escalado(X_train: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Media y desviación de train para estandarizar (z-score).

    Solo escala las columnas de COLUMNAS_A_ESCALAR presentes en X_train.
    """
    columnas = [c for c in COLUMNAS_A_ESCALAR if c in X_train.columns]
    return {c: (float(X_train[c].mean()), float(X_train[c].std(ddof=0))) for c in columnas}


def aplicar_escalado(X: pd.DataFrame, parametros: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """Estandariza las columnas indicadas con la media y desviación de train."""
    X = X.copy()
    for columna, (media, desviacion) in parametros.items():
        X[columna] = (X[columna] - media) / desviacion
    return X


# ---------------------------------------------------------------------------
# 5. Guardado
# ---------------------------------------------------------------------------

def guardar_datasets(df_limpio: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame,
                     train_esc: pd.DataFrame, test_esc: pd.DataFrame) -> list[Path]:
    """Guarda los CSV en sus carpetas de data/ (train y test ya incluyen el objetivo).

    * data/modificado/    -> saber11_limpio.csv
    * data/entrenamiento/ -> train.csv y train_escalado.csv
    * data/test/          -> test.csv y test_escalado.csv
    """
    destinos = {
        CARPETA_MODIFICADO / "saber11_limpio.csv": df_limpio,
        CARPETA_ENTRENAMIENTO / "train.csv": train,
        CARPETA_ENTRENAMIENTO / "train_escalado.csv": train_esc,
        CARPETA_TEST / "test.csv": test,
        CARPETA_TEST / "test_escalado.csv": test_esc,
    }
    for ruta, tabla in destinos.items():
        ruta.parent.mkdir(parents=True, exist_ok=True)
        tabla.to_csv(ruta, index=False, encoding="utf-8-sig")
    return list(destinos)
