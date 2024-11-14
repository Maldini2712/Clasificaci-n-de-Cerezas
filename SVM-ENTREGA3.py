# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 22:12:47 2024

@author: maldi
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns 

datos = pd.read_csv('CEREZASOFICIAL.csv', encoding='latin-1', sep=';')

# Convertir la columna 'Rentabilidad' a numérica, forzando errores a NaN
datos['Rentabilidad'] = pd.to_numeric(datos['Rentabilidad'], errors='coerce')

#to_get_dummies_for = ['Pais de Destino', 'Tipo de Bulto', 'Nave']
#datos = pd.get_dummies(data = datos, columns = to_get_dummies_for, drop_first = True)

# Seleccionar los atributos 
atributos_seleccionados = ['C+Otros','TotalExp'] #+ [col for col in datos.columns if 'Pais de Destino_' in col or 'Tipo de Bulto_' in col or 'Nave_' in col]
X = datos[atributos_seleccionados]
Y = datos['Rentabilidad'].dropna()

y_categorico = pd.qcut(Y, q=3, labels=['Baja', 'Media', 'Alta'])

datos_completos = pd.concat([X, y_categorico], axis=1).dropna()
X = datos_completos[atributos_seleccionados]
Y = datos_completos['Rentabilidad']

print("Número de muestras en X:", X.shape[0])
print("Número de muestras en y:", Y.shape[0])

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, Y,
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
    
clf = SVC(kernel='poly').fit(X_train, y_train)

print("Entreno")
y_pred_train = clf.predict(X_train)
metrics_score(y_train, y_pred_train)

# Evaluar el modelo
print("Prueba")
y_pred_test = clf.predict(X_test)
metrics_score(y_test, y_pred_test)

print("Precisión del modelo:", clf.score(X_test, y_test))

# Imprimir un reporte de clasificación
print(classification_report(y_test, y_pred_test, target_names=['Baja', 'Media', 'Alta']))