archivo="Consumo_cerveza.csv"

import numpy as np
import pandas as pd


import plotly.express as px
import plotly.graph_objects as go


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from datetime import datetime

formato = lambda x: datetime.strptime(x, "%d/%m/%Y")

datos=pd.read_csv(archivo, parse_dates=['Fecha'])

datos=pd.read_csv(archivo)


print(datos["Temperatura Media (C)"][0])


print("El dataset contiene {} filas y {} columnas".format(datos.shape[0], datos.shape[1]))

datos.info()

datos.head()
datos.columns


datos.isnull().sum()

datos =  datos.dropna()



datos.columns = ['fecha', 'Temp_Media', 'Temp_Min', 'Temp_Max', 'Precipitacion ', 'Finde', 'Consumo_Litros']

dias_entre_semana = sum(datos[datos.Finde == 0]['Consumo_Litros'])
fin_de_semana = sum(datos[datos.Finde==1]['Consumo_Litros'])

etiqueta = ['Dias entre semana','Fin de semana']
valores = [dias_entre_semana, fin_de_semana]
colores = ['crimson']

fig = go.Figure(data=[go.Bar(x=etiqueta, y=valores, marker_color= colores)])
fig.show()



datos['fecha'] = pd.to_datetime(datos['fecha'], format="%d/%m/%y")



datos['Mes'] = datos['fecha'].apply(lambda x: x.strftime('%B'))
datos['Dia'] = datos['fecha'].apply(lambda x: x.strftime('%A'))

datos.head()



figura = px.box(datos, x="Dia", y="Consumo_Litros", color="Dia", orientation='v', notched=True, title = 'Consumo de cerveza por día de la semana' )


figura.show()



figura = px.box(datos, x="Mes", y="Consumo_Litros", color="Mes", orientation='v', notched=True, title = 'Consumo de cerveza por mes del año' )

figura.show()


