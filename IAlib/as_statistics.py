# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 12:43:10 2024

@author: adria
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 14:30:31 2024

@author: adria
"""

import numpy as np
import pandas as pd
from scipy import stats

def calcular_media(df, columna):
    """
    Calcula la media de una columna en un DataFrame.
    
    Parámetros:
    - df: pd.DataFrame
    - columna: str
      Nombre de la columna para calcular la media.
    
    Retorna:
    - float: Media de la columna.
    """
    if columna in df.columns:
        return df[columna].mean()
    else:
        raise ValueError(f"La columna '{columna}' no existe en el DataFrame.")

def calcular_mediana(df, columna):
    """
    Calcula la mediana de una columna en un DataFrame.
    
    Parámetros:
    - df: pd.DataFrame
    - columna: str
      Nombre de la columna para calcular la mediana.
    
    Retorna:
    - float: Mediana de la columna.
    """
    if columna in df.columns:
        return df[columna].median()
    else:
        raise ValueError(f"La columna '{columna}' no existe en el DataFrame.")

def calcular_moda(df, columna):
    """
    Calcula la moda de una columna en un DataFrame.
    
    Parámetros:
    - df: pd.DataFrame
    - columna: str
      Nombre de la columna para calcular la moda.
    
    Retorna:
    - float: Moda de la columna.
    """
    if columna in df.columns:
        return df[columna].mode()[0]
    else:
        raise ValueError(f"La columna '{columna}' no existe en el DataFrame.")

def calcular_desviacion_estandar(df, columna):
    """
    Calcula la desviación estándar de una columna en un DataFrame.
    
    Parámetros:
    - df: pd.DataFrame
    - columna: str
      Nombre de la columna para calcular la desviación estándar.
    
    Retorna:
    - float: Desviación estándar de la columna.
    """
    if columna in df.columns:
        return df[columna].std()
    else:
        raise ValueError(f"La columna '{columna}' no existe en el DataFrame.")

def calcular_correlacion(df, columna1, columna2):
    """
    Calcula la correlación entre dos columnas en un DataFrame.
    
    Parámetros:
    - df: pd.DataFrame
    - columna1: str
      Nombre de la primera columna.
    - columna2: str
      Nombre de la segunda columna.
    
    Retorna:
    - float: Coeficiente de correlación entre las dos columnas.
    """
    if columna1 in df.columns and columna2 in df.columns:
        return df[[columna1, columna2]].corr().iloc[0, 1]
    else:
        raise ValueError(f"Una o ambas columnas '{columna1}' y '{columna2}' no existen en el DataFrame.")

def calcular_zscore(df, columna):
    """
    Calcula el z-score de una columna en un DataFrame.
    
    Parámetros:
    - df: pd.DataFrame
    - columna: str
      Nombre de la columna para calcular el z-score.
    
    Retorna:
    - pd.Series: Z-scores de la columna.
    """
    if columna in df.columns:
        return (df[columna] - df[columna].mean()) / df[columna].std()
    else:
        raise ValueError(f"La columna '{columna}' no existe en el DataFrame.")

def calcular_iqr(df, columna):
    """
    Calcula el rango intercuartil (IQR) de una columna en un DataFrame.
    
    Parámetros:
    - df: pd.DataFrame
    - columna: str
      Nombre de la columna para calcular el IQR.
    
    Retorna:
    - float: Rango intercuartil de la columna.
    """
    if columna in df.columns:
        Q1 = df[columna].quantile(0.25)
        Q3 = df[columna].quantile(0.75)
        return Q3 - Q1
    else:
        raise ValueError(f"La columna '{columna}' no existe en el DataFrame.")
