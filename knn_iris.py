archivo="C:/Temp/ds/iris.csv"

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split

datos=pd.read_csv(archivo)

datos.groupby('species').size()

datos.head()

grafica = px.scatter(datos, x="sepal_length", y="sepal_width", color="species")

grafica.update_traces(marker=dict(size=12,line=dict(width=2,color='DarkSlateGrey')),selector=dict(mode='markers'))

grafica.show()

grafica = px.scatter(datos, x="petal_length", y="petal_width", color="species")

grafica.update_traces(marker=dict(size=12,line=dict(width=2,color='DarkSlateGrey')),selector=dict(mode='markers'))

grafica.show()

grafica = px.scatter(datos, x="sepal_length", y="sepal_width",  facet_col="species")
grafica.update_xaxes(title_font=dict(size=18, family='Courier', color='crimson'))
grafica.update_yaxes(title_font=dict(size=18, family='Courier', color='crimson'))

grafica.show()

X = datos.iloc[:, :-1].values

y = datos.iloc[:, 4].values

X_entreno, X_prueba, y_entreno, y_prueba = train_test_split(X, y, test_size=0.2) 

print('Hay {} muestras en el conjunto de entrenamiento y {} muestras en el conjunto de pruebas (validación)'.format(X_entreno.shape[0], X_prueba.shape[0]))

trace_specs = [
    [X_entreno, y_entreno, 'virginica', 'Entreno', 'square'],
    [X_entreno, y_entreno, 'versicolor', 'Entreno', 'circle'],
    [X_entreno, y_entreno, 'setosa', 'Entreno', 'diamond'],
    [X_prueba, y_prueba, 'virginica', 'Prueba', 'square-dot'],
    [X_prueba, y_prueba, 'versicolor', 'Prueba', 'circle-dot'],
    [X_prueba, y_prueba, 'setosa', 'Prueba', 'diamond-dot']
]

grafica2 = go.Figure(data=[
    go.Scatter(
        x=X[y==label, 0], y=X[y==label, 1],
        name=f'{split} , {label}',
        mode='markers', marker_symbol=marker
    )
    for X, y, label, split, marker in trace_specs
])

grafica2.update_traces(
    marker_size=12, marker_line_width=1.5,
    marker_color="lightyellow"
)

grafica2.show()


from sklearn.neighbors import KNeighborsClassifier

error = []

for i in range(1, 30):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_entreno, y_entreno)
    pred_i = knn.predict(X_prueba)
    error.append(np.mean(pred_i != y_prueba))


grafica4 = px.line(x=range(1, 30),y=error,  markers=True)

grafica4.show()


from sklearn.metrics import accuracy_score

clf = KNeighborsClassifier(n_neighbors=5)

clf.fit(X_entreno, y_entreno)

y_prediccion = clf.predict(X_prueba)

precision = accuracy_score(y_prueba, y_prediccion)*100

print('La precisión de nuestro modelo es: ' + str(round(precision, 2)) + ' %.')

from sklearn.metrics import classification_report
print(classification_report(y_prueba, y_prediccion))


valor_clasificar=[[5.7, 3.0 , 4.2, 1.2],[5.1, 3.8, 1.5, 0.3]]

prediccion_knn= clf.predict(valor_clasificar)

prediccion_knn



from sklearn.cluster import KMeans

marca = []

for i in range(1, 12):
    km = KMeans(n_clusters=i)
    km.fit(X_entreno)
    marca.append(km.inertia_)


grafica5 = px.line(x=range(1, 12),y=marca,  markers=True)

grafica5.show()

ncluster = 3

semilla = 0

km = KMeans(n_clusters=ncluster, random_state=semilla)
km.fit(X_entreno)


y_cluster_kmeans = km.predict(X_entreno)
y_cluster_kmeans

resultado=pd.DataFrame(X_entreno,columns=datos.iloc[:, :-1].columns)
resultado['Especies']=y_entreno

resultado['ClusterKmeans'] = y_cluster_kmeans

resultado.head(20)

from sklearn.decomposition import PCA

ndimensions = 2



pca = PCA(n_components=ndimensions, random_state=42)
pca.fit(X_entreno)
X_pca_array = pca.transform(X_entreno)
X_pca = pd.DataFrame(X_pca_array, columns=['PC1','PC2'])

X_pca.sample(5)

y_id_array = pd.Categorical(y_entreno).codes


resultado2 = X_pca.copy()
resultado2['ClusterKmeans'] = y_cluster_kmeans

resultado2['SpeciesId'] = y_id_array

resultado2.sample(20)
