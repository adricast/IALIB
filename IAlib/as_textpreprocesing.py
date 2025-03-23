# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 12:55:52 2024

@author: adria
"""

import string
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk import download

# Descargar recursos de NLTK si no están presentes
download('punkt')
download('stopwords')
download('wordnet')

# Inicializar lematizadores y stopwords
lemmatizer_en = WordNetLemmatizer()
stop_words_en = set(stopwords.words('english'))
stop_words_es = set(stopwords.words('spanish'))


def preprocess_text(text, language='spanish'):
    """
    Preprocesa el texto: convierte a minúsculas, elimina puntuaciones, 
    elimina stopwords y realiza lematización o stemming. Compatible con inglés y español.
    """
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))

    stop_words = stop_words_es if language == 'spanish' else stop_words_en
    
    if language == 'spanish':
        stemmer = SnowballStemmer('spanish')
        tokens = word_tokenize(text, language='spanish')
        tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    else:
        tokens = word_tokenize(text, language='english')
        tokens = [lemmatizer_en.lemmatize(token) for token in tokens if token not in stop_words]
    
    return ' '.join(tokens)


def preprocess_rnn(text, language='english'):
    """
    Preprocesamiento para modelos RNN: elimina caracteres especiales, tokeniza y lematiza.
    
    Parámetros:
    - text: str
      Texto a procesar.
    - language: str, por defecto 'english'
      Idioma del texto ('english' o 'spanish').

    Retorna:
    - str: Texto preprocesado para modelos RNN.
    """
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    tokens = word_tokenize(text, language=language)
    
    stop_words = stop_words_en if language == 'english' else stop_words_es
    
    if language == 'english':
        tokens = [lemmatizer_en.lemmatize(token) for token in tokens if token not in stop_words]
    else:
        stemmer = SnowballStemmer('spanish')
        tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    
    return ' '.join(tokens)


def remove_punctuation(text):
    """Elimina la puntuación de un texto."""
    return text.translate(str.maketrans('', '', string.punctuation))


def remove_stopwords(text, language='spanish'):
    """Elimina las stopwords de un texto."""
    stop_words = stop_words_es if language == 'spanish' else stop_words_en
    tokens = word_tokenize(text, language=language)
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)


def lemmatize_text(text):
    """Lematiza un texto usando WordNetLemmatizer (solo inglés)."""
    tokens = word_tokenize(text, language='english')
    tokens = [lemmatizer_en.lemmatize(token) for token in tokens]
    return ' '.join(tokens)


def tokenize_text(text, language='english'):
    """Tokeniza un texto en palabras."""
    return word_tokenize(text, language=language)
