# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 12:30:31 2024

@author: adria
"""

# IAlib/datapreprocessing.py

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from scipy.stats import zscore
from as_statistics import calcular_media, calcular_mediana, calcular_moda, calcular_desviacion_estandar, calcular_iqr
from as_textpreprocesing import preprocess_text, remove_punctuation, remove_stopwords, lemmatize_text
import re
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def rellenar_valores_faltantes(df, estrategia='media', columnas=None):
    """
    Rellena los valores faltantes en un DataFrame.
    
    Parámetros:
    - df: pd.DataFrame
    - estrategia: str, por defecto 'media'
      La estrategia para rellenar los valores faltantes, uno de {'media', 'mediana', 'modo'}.
    - columnas: list, por defecto None
      Lista de columnas a rellenar. Si es None, se rellenan todas las columnas.
    
    Retorna:
    - pd.DataFrame: DataFrame con los valores faltantes rellenados.
    """
    if columnas is None:
        columnas = df.columns
    
    for col in columnas:
        if estrategia == 'media':
            df[col].fillna(calcular_media(df, col), inplace=True)
        elif estrategia == 'mediana':
            df[col].fillna(calcular_mediana(df, col), inplace=True)
        elif estrategia == 'modo':
            df[col].fillna(calcular_moda(df, col), inplace=True)
    
    return df

def normalizar_datos(df, columnas=None, rango_caracteristicas=(0, 1)):
    """
    Normaliza los datos en un DataFrame.
    
    Parámetros:
    - df: pd.DataFrame
    - columnas: list, por defecto None
      Lista de columnas a normalizar. Si es None, se normalizan todas las columnas.
    - rango_caracteristicas: tuple, por defecto (0, 1)
      Rango deseado de los datos transformados.
    
    Retorna:
    - pd.DataFrame: DataFrame con los valores normalizados.
    """
    scaler = MinMaxScaler(feature_range=rango_caracteristicas)
    
    if columnas is None:
        columnas = df.columns
    
    df[columnas] = scaler.fit_transform(df[columnas])
    
    return df

def normalizar_zscore(df, columnas=None):
    """
    Normaliza los datos en un DataFrame utilizando la escala Z.
    
    Parámetros:
    - df: pd.DataFrame
    - columnas: list, por defecto None
      Lista de columnas a normalizar. Si es None, se normalizan todas las columnas.
    
    Retorna:
    - pd.DataFrame: DataFrame con los valores normalizados.
    """
    scaler = StandardScaler()
    
    if columnas is None:
        columnas = df.columns
    
    df[columnas] = scaler.fit_transform(df[columnas])
    
    return df

def codificar_categoricas(df, columnas, tipo_codificacion='etiqueta'):
    """
    Codifica las variables categóricas en un DataFrame.
    
    Parámetros:
    - df: pd.DataFrame
    - columnas: list
      Lista de columnas categóricas a codificar.
    - tipo_codificacion: str, por defecto 'etiqueta'
      Tipo de codificación, ya sea 'etiqueta' o 'onehot'.
    
    Retorna:
    - pd.DataFrame: DataFrame con las variables categóricas codificadas.
    """
    if tipo_codificacion == 'etiqueta':
        for col in columnas:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    elif tipo_codificacion == 'onehot':
        df = pd.get_dummies(df, columns=columnas)
    
    return df

def eliminar_outliers(df, columnas, metodo='zscore', umbral=3):
    """
    Elimina los outliers del DataFrame basándose en el método especificado.
    
    Parámetros:
    - df: pd.DataFrame
    - columnas: list
      Lista de columnas para verificar outliers.
    - metodo: str, por defecto 'zscore'
      Método para detectar outliers, ya sea 'zscore' o 'iqr'.
    - umbral: float, por defecto 3
      Umbral para la detección de outliers.
    
    Retorna:
    - pd.DataFrame: DataFrame con los outliers eliminados.
    """
    if metodo == 'zscore':
        for col in columnas:
            df = df[(zscore(df[col].dropna()) < umbral)]
    elif metodo == 'iqr':
        for col in columnas:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df = df[~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))]
    
    return df

def imputar_knn(df, columnas=None, n_neighbors=5):
    """
    Imputa los valores faltantes utilizando el algoritmo KNN.
    
    Parámetros:
    - df: pd.DataFrame
    - columnas: list, por defecto None
      Lista de columnas a imputar. Si es None, se imputan todas las columnas.
    - n_neighbors: int, por defecto 5
      Número de vecinos a considerar para la imputación.
    
    Retorna:
    - pd.DataFrame: DataFrame con los valores imputados.
    """
    imputer = KNNImputer(n_neighbors=n_neighbors)
    
    if columnas is None:
        columnas = df.columns
    
    df[columnas] = imputer.fit_transform(df[columnas])
    
    return df

def analizar_distribucion(df, columnas=None):
    """
    Visualiza la distribución de los datos en un DataFrame.
    
    Parámetros:
    - df: pd.DataFrame
    - columnas: list, por defecto None
      Lista de columnas a analizar. Si es None, se analizan todas las columnas.
    
    Retorna:
    - None
    """
    if columnas is None:
        columnas = df.columns
    
    for col in columnas:
        plt.figure(figsize=(10, 4))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f'Distribución de {col}')
        plt.show()

def generar_reporte(df):
    """
    Genera un informe con estadísticas descriptivas y resumen de transformaciones.
    
    Parámetros:
    - df: pd.DataFrame
    
    Retorna:
    - dict: Informe con estadísticas descriptivas.
    """
    reporte = {
        'estadisticas': df.describe().to_dict(),
        'columnas': df.columns.tolist(),
        'tipos_de_datos': df.dtypes.to_dict()
    }
    
    return reporte

def transformar_fechas(df, columnas, formato='%Y-%m-%d'):
    """
    Convierte las columnas de fecha a un formato específico.
    
    Parámetros:
    - df: pd.DataFrame
    - columnas: list
      Lista de columnas de fecha.
    - formato: str, por defecto '%Y-%m-%d'
      Formato de fecha deseado.
    
    Retorna:
    - pd.DataFrame: DataFrame con las fechas transformadas.
    """
    for col in columnas:
        df[col] = pd.to_datetime(df[col], format=formato)
    
    return df

def aplicar_preprocesamiento_texto(df, columna_texto, language='spanish'):
    """
    Aplica el preprocesamiento de texto a una columna de un DataFrame.
    
    Parámetros:
    - df: pd.DataFrame
    - columna_texto: str
      Nombre de la columna de texto a procesar.
    - language: str, por defecto 'spanish'
      Idioma para el preprocesamiento del texto.
    
    Retorna:
    - pd.DataFrame: DataFrame con la columna de texto preprocesada.
    """
    df[columna_texto] = df[columna_texto].apply(lambda x: preprocess_text(x, language=language))
    return df

def eliminar_urls(texto):
    """
    Elimina URLs del texto.
    """
    return re.sub(r'http\S+|www\S+|https\S+', '', texto, flags=re.MULTILINE)

def eliminar_menciones(texto):
    """
    Elimina menciones de redes sociales (por ejemplo, @usuario).
    """
    return re.sub(r'@\w+', '', texto)

def corregir_ortografia(texto):
    """
    Corrige errores ortográficos en el texto.
    """
    return str(TextBlob(texto).correct())
