# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 10:54:39 2024

@author: maldi
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import learning_curve
import numpy as np


datos = pd.read_csv('CEREZASOFICIAL.csv', encoding='latin-1', sep=';')

# Convertir la columna 'Rentabilidad' a numérica, forzando errores a NaN
datos['Rentabilidad'] = pd.to_numeric(datos['Rentabilidad'], errors='coerce')

to_get_dummies_for = ['Pais de Destino', 'Tipo de Bulto', 'Nave']
datos = pd.get_dummies(data = datos, columns = to_get_dummies_for, drop_first = True)


# Seleccionar los atributos 
atributos_seleccionados = ['TotalExp'] + [col for col in datos.columns if 'Pais de Destino_' in col or 'Tipo de Bulto_' in col or 'Nave_' in col]
X = datos[atributos_seleccionados]
y = datos['Rentabilidad'].dropna()

# Crear las categorías de rentabilidad baja, media, alta
y_categorico = pd.qcut(y, q=3, labels=['Baja', 'Media', 'Alta'])

# Combinar X e y y eliminar las filas con NaN
datos_completos = pd.concat([X, y_categorico], axis=1).dropna()
X = datos_completos[atributos_seleccionados]
y = datos_completos['Rentabilidad']

# Imprimir el tamaño de X e y
print("Número de muestras en X:", X.shape[0])
print("Número de muestras en y:", y.shape[0])

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.15,
                                                    random_state=42)

# Estandarizar 
sc = StandardScaler()
X_train = X_train.replace({'\.': '', ',': '.'}, regex=True).astype(float)
X_test = X_test.replace({'\.': '', ',': '.'}, regex=True).astype(float)
X_train_array = sc.fit_transform(X_train.values)
X_test_array = sc.transform(X_test.values)
X_train = pd.DataFrame(X_train_array, index=X_train.index, columns=X_train.columns)
X_test = pd.DataFrame(X_test_array, index=X_test.index, columns=X_test.columns)

def metrics_score(actual, predicted):
    print(classification_report(actual, predicted))
    
    cm = confusion_matrix(actual, predicted)
    plt.figure(figsize=(8,5))

    # Actualizar etiquetas para las categorías de rentabilidad
    sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=['Baja', 'Media', 'Alta'], yticklabels=['Baja', 'Media', 'Alta'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()


dt = DecisionTreeClassifier(class_weight = 'balanced', random_state = 1)

dt.fit(X_train, y_train)

y_train_pred = dt.predict(X_train)
metrics_score(y_train, y_train_pred)

y_test_pred = dt.predict(X_test)
metrics_score(y_test, y_test_pred)


importances = dt.feature_importances_

columns = X.columns

importance_df = pd.DataFrame(importances, index = columns, columns = ['Importance']).sort_values(by = 'Importance', ascending = False)

plt.figure(figsize = (13, 13))

sns.barplot(x = importance_df.Importance, y = importance_df.index)

features = list(X.columns)

plt.figure(figsize = (30, 20))

tree.plot_tree(dt, max_depth = 4, feature_names = features, filled = True, fontsize = 12, node_ids = True, class_names = ['Baja', 'Media', 'Alta'])

plt.show()



