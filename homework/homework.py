import os
import gzip
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix

def cargar_y_procesar_datos(ruta):
    df = pd.read_csv(ruta, compression="zip")
    df.rename(columns={'default payment next month': 'default'}, inplace=True)
    df.drop(columns=['ID'], inplace=True, errors='ignore')
    df.dropna(inplace=True)
    df = df[(df["EDUCATION"] != 0) & (df["MARRIAGE"] != 0)]
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: 4 if x > 4 else x)
    return df

def definir_pipeline():
    columnas_categoricas = ["SEX", "EDUCATION", "MARRIAGE"]
    transformador_columnas = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), columnas_categoricas)
    ], remainder='passthrough')
    modelo = Pipeline([
        ('preprocesador', transformador_columnas),
        ('escalador', MinMaxScaler()),
        ('selector_caracteristicas', SelectKBest(score_func=f_classif, k=10)),
        ('clasificador', LogisticRegression(max_iter=500, random_state=42))
    ])
    return modelo

def optimizar_hiperparametros(modelo, x_train, y_train):
    hiperparametros = {
        'selector_caracteristicas__k': range(1, 11),
        'clasificador__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'clasificador__penalty': ['l1', 'l2'],
        'clasificador__solver': ['liblinear'],
        "clasificador__max_iter": [100, 200]
    }
    busqueda = GridSearchCV(modelo, param_grid=hiperparametros, cv=10, scoring='balanced_accuracy', n_jobs=-1, refit=True)
    busqueda.fit(x_train, y_train)
    return busqueda

def guardar_modelo(modelo, ruta="files/models/model.pkl.gz"):
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with gzip.open(ruta, 'wb') as archivo:
        pickle.dump(modelo, archivo)

def calcular_metricas(y_real, y_predicho, conjunto):
    return {
        'type': 'metrics',
        'dataset': conjunto,
        'precision': precision_score(y_real, y_predicho),
        'balanced_accuracy': balanced_accuracy_score(y_real, y_predicho),
        'recall': recall_score(y_real, y_predicho),
        'f1_score': f1_score(y_real, y_predicho)
    }

def calcular_matriz_confusion(y_real, y_predicho, conjunto):
    matriz = confusion_matrix(y_real, y_predicho)
    return {
        'type': 'cm_matrix',
        'dataset': conjunto,
        'true_0': {"predicted_0": int(matriz[0, 0]), "predicted_1": int(matriz[0, 1])},
        'true_1': {"predicted_0": int(matriz[1, 0]), "predicted_1": int(matriz[1, 1])}
    }

def guardar_metricas(resultados, ruta="files/output/metrics.json"):
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "w") as archivo:
        for resultado in resultados:
            archivo.write(json.dumps(resultado) + "\n")

# Carga y procesamiento de datos
train_df = cargar_y_procesar_datos("files/input/train_data.csv.zip")
test_df = cargar_y_procesar_datos("files/input/test_data.csv.zip")

x_train, y_train = train_df.drop(columns=['default']), train_df['default']
x_test, y_test = test_df.drop(columns=['default']), test_df['default']

# Construcción del modelo
modelo = definir_pipeline()
modelo_optimizado = optimizar_hiperparametros(modelo, x_train, y_train)

# Guardar modelo
guardar_modelo(modelo_optimizado)

# Evaluar el modelo
y_train_pred = modelo_optimizado.predict(x_train)
y_test_pred = modelo_optimizado.predict(x_test)

resultados_metricas = [
    calcular_metricas(y_train, y_train_pred, 'train'),
    calcular_metricas(y_test, y_test_pred, 'test'),
    calcular_matriz_confusion(y_train, y_train_pred, 'train'),
    calcular_matriz_confusion(y_test, y_test_pred, 'test')
]

# Guardar métricas
guardar_metricas(resultados_metricas)
