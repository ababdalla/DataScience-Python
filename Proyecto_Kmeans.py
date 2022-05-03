archivo="C:/Personal/Home/New folder/Maestria/4_Introducción a la minería de datos/Reprasentative Sample of Bitcoin Blockchain Data.csv"

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split

datos=pd.read_csv(archivo)

datos.head()

datos.columns

datos.isnull().any().any()
datos.isnull().sum()
datos

# Se evalúan varias gráficas para observar el comportamiento general de cada columna
# Grafica 1 Valor_Satoshi vs Tiempo
grafica = px.scatter(datos, x="Block_Time", y="Value_Sat")
grafica.update_traces(marker=dict(size=12,line=dict(width=2,color='DarkSlateGrey')),selector=dict(mode='markers'))
grafica.show()
##grafica.write_image("figure.png", engine="kaleido")

# Grafica 2 Valor_Bitcoin vs Tiempo
# Se observa que es lo mismo que la gráfica anterior
grafica2 = px.scatter(datos, x="Block_Time", y="Value_BTC")
grafica2.update_traces(marker=dict(size=12,line=dict(width=2,color='DarkSlateGrey')),selector=dict(mode='markers'))
grafica2.show()

# Grafica 3 Recompensa vs Tiempo
grafica3 = px.scatter(datos, x="Block_Time", y="Block_Reward_BTC")
grafica3.update_traces(marker=dict(size=12,line=dict(width=2,color='DarkSlateGrey')),selector=dict(mode='markers'))
grafica3.show()

# Grafica 4 Comisiones vs Tiempo
grafica4 = px.scatter(datos, x="Block_Time", y="Transaction_Fees_BTC")
grafica4.update_traces(marker=dict(size=12,line=dict(width=2,color='DarkSlateGrey')),selector=dict(mode='markers'))
grafica4.show()

# Grafica 5 Valor_bitcoin vs Precio
# Esta gráfica es clave para clasificación, dado que nos muestra segmentos
# en los cuales se podría clasificar alta, media o baja ganancia.
grafica5 = px.scatter(datos, x="Value_BTC", y="Price")
grafica5.update_traces(marker=dict(size=12,line=dict(width=2,color='DarkSlateGrey')),selector=dict(mode='markers'))
grafica5.show()

# Grafica 6 valor BTC vs Recompensa
# Esta gráfica muestra que que value BTC incluye tanto el reward como el fee
grafica6 = px.scatter(datos, x="Block_Reward_BTC", y="Value_BTC")
grafica6.update_traces(marker=dict(size=12,line=dict(width=2,color='DarkSlateGrey')),selector=dict(mode='markers'))
grafica6.show()

# Gráfica 7 Hash_Rate vs Difficulty
# Esta gráfica nos muestra que es posible realizar una regresión lineal para poder
# determinar cuanto Hash Rate necesitamos en nuestros equipos para una determinada
# dificultad de bloques.
grafica7 = px.scatter(datos, x="Difficulty", y="Hash_Rate")
grafica7.update_traces(marker=dict(size=12,line=dict(width=2,color='DarkSlateGrey')),selector=dict(mode='markers'))
grafica7.show()

# Limpieza de Datos

del datos['Block_Hash']
del datos['Block_Height']
del datos['Is_Coinbase?']
del datos['Block_Time']

# Verificación de datos
datos.columns
datos.head(10)  

# Tomamos las columnas que servirán para split test / train
X = datos
columnas= ['Price']
y = datos[columnas].values

X_entreno, X_prueba, y_entreno, y_prueba = train_test_split(X, y, test_size=0.2) 

print('Hay {} muestras en el conjunto de entrenamiento y {} muestras en el conjunto de pruebas (validaciÃ³n)'.format(X_entreno.shape[0], X_prueba.shape[0]))

# Importamos la libreria para KMeans
from sklearn.cluster import KMeans

# Generamos los errores para diferente numero de clusters
# y graficamos para generar el codo de Jambú
marca = []
for i in range(1, 12):
    km = KMeans(n_clusters=i)
    km.fit(X_entreno)
    marca.append(km.inertia_)

grafica8 = px.line(x=range(1, 12),y=marca,  markers=True)
grafica8.show()

# De la gráfica observamos que con 3 clusters obtendríamos la clasificación óptima
ncluster = 3
# Indicamos que no usaremos inicio aleatorio
semilla = 0
# Ejecutamos la función y entrenamos el modelo
km = KMeans(n_clusters=ncluster, random_state=semilla)
km.fit(X_entreno)

# Utilizamos los datos de entreno para predecir usando el dato de prueba
y_cluster_kmeans = km.predict(X_entreno)
y_cluster_kmeans

# Generamos un nuevo dataset para visualizar el resultado de la clasificación
resultado=pd.DataFrame(X_entreno,columns=datos.columns)
resultado['ClusterKmeans'] = y_cluster_kmeans
resultado.head(20)

grafica9 = px.scatter(resultado, x="Value_BTC", y="Price", color='ClusterKmeans')
grafica9.update_traces(marker=dict(size=12,line=dict(width=2,color='DarkSlateGrey')),selector=dict(mode='markers'))
grafica9.show()
