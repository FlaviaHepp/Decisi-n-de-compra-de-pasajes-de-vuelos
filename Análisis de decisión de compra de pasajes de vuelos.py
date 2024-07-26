"""
Este conjunto de datos contiene datos de tarifas de vuelos que se recopilaron del sitio web de EaseMyTrip mediante técnicas de raspado web. Los 
datos se recopilaron con el objetivo de proporcionar a los usuarios información que pudiera ayudarles a tomar decisiones informadas sobre cuándo 
y dónde comprar billetes de avión. Al analizar los patrones en las tarifas de los vuelos a lo largo del tiempo, los usuarios pueden identificar 
los mejores momentos para reservar boletos y potencialmente ahorrar dinero.

Fuentes:

Datos recopilados mediante script Python con las bibliotecas Beautiful Soup y Selenium.
El script recopiló datos sobre varios detalles del vuelo, como fecha de reserva, fecha de viaje, aerolínea y clase, hora y origen de salida, hora 
y destino de llegada, duración, escalas totales y precio.
El proceso de raspado se diseñó para recopilar datos de vuelos que salen de un conjunto específico de aeropuertos (los 7 aeropuertos más 
transitados de la India).
Tenga en cuenta que la función Hora de salida también incluye el aeropuerto de origen y la función Hora de llegada también incluye el aeropuerto 
de destino. Que luego se extrae en Cleaned_dataset. Además, se han proporcionado conjuntos de datos tanto limpios como extraídos para que uno 
pueda utilizar el conjunto de datos según sus necesidades y conveniencia.
Inspiración:

Conjunto de datos creado para proporcionar a los usuarios un recurso valioso para analizar tarifas de vuelos en la India.
La información detallada sobre las tarifas de los vuelos a lo largo del tiempo se puede utilizar para desarrollar modelos de precios más precisos 
e informar a los usuarios sobre los mejores momentos para reservar boletos.
Los datos también se pueden utilizar para estudiar tendencias y patrones en la industria de viajes a través del aire y pueden actuar como un 
recurso valioso para investigadores y analistas.
Limitaciones:

Este conjunto de datos solo cubre vuelos que salen de aeropuertos específicos y se limitan a un período de tiempo determinado.
Para realizar un análisis de series de tiempo, se deben recopilar datos de al menos los 10 aeropuertos más transitados durante 365 días.
Esto no cubre las variaciones en los precios del combustible de aviación, ya que es el factor que influye a la hora de decidir la tarifa, por lo 
que el mismo conjunto de datos podría no ser útil para el próximo año, pero intentaré actualizarlo dos veces al año.
Además, la oferta y la demanda para un asiento de vuelo en particular no están disponibles en el conjunto de datos, ya que estos datos no están 
disponibles públicamente en ningún sitio web de reserva de vuelos.
Alcance de la mejora:

El conjunto de datos podría mejorarse incluyendo características adicionales como los precios actuales del combustible de aviación y la distancia 
entre el origen y el destino en términos de longitud y latitud.
Los datos también podrían ampliarse para incluir más aerolíneas y más aeropuertos, proporcionando una visión más completa del mercado de vuelos.
Además, puede resultar útil incluir datos sobre cancelaciones de vuelos, retrasos y otros factores que pueden afectar el precio y la disponibilidad 
de los vuelos.
Finalmente, si bien el conjunto de datos actual proporciona información sobre los precios de los vuelos, no incluye información sobre la calidad 
de la experiencia del vuelo, como el espacio para las piernas, las comodidades a bordo y las opiniones de los clientes. Incluir este tipo de datos 
podría proporcionar una imagen más completa del mercado de vuelos y ayudar a los viajeros a tomar decisiones más informadas.

Una aerolínea es una empresa que brinda servicios de transporte aéreo de pasajeros y carga. Las aerolíneas utilizan aviones para prestar estos 
servicios y pueden formar asociaciones o alianzas con otras aerolíneas para acuerdos de código compartido, en los que ambas ofrecen y operan el 
mismo vuelo. Generalmente, las compañías aéreas son reconocidas con un certificado o licencia de operación aérea emitido por un organismo 
gubernamental de aviación. Las aerolíneas pueden ser operadores regulares o chárter.

Las aerolíneas asignan precios a sus servicios en un intento de maximizar la rentabilidad. El precio de los billetes de avión se ha vuelto cada 
vez más complicado a lo largo de los años y ahora está determinado en gran medida por sistemas informatizados de gestión del rendimiento.

El precio de un billete de avión se ve afectado por una serie de factores, como la duración del vuelo, los días que quedan para la salida, la 
hora de llegada y de salida, etc. Las organizaciones aéreas pueden reducir el coste en el momento en que necesitan construir el mercado y en el 
momento cuando las entradas son menos accesibles. Pueden maximizar los costos. El precio puede depender de diferentes factores. Cada factor tiene 
sus propias reglas y algoritmos para fijar el precio en consecuencia. Los recientes avances en Inteligencia Artificial (IA) y Aprendizaje 
Automático (ML) permiten inferir dichas reglas y modelar la variación de precios."""

# Importando todas las bibliotecas requeridas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from tqdm import tqdm_notebook
import warnings
warnings.filterwarnings('ignore')
# Preprocesamiento de datos
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
# Selección de características
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_regression, SelectKBest, mutual_info_regression
# Modelos
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import GradientBoostingRegressor as GBR
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
# Métrica 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# Construya los modelos de regresión/regresor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
#Importando las bibliotecas requeridas 
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.metrics import r2_score
#XGBoost Regresión
import xgboost as xgb


# Veamos qué hay en los datos
df = pd.read_csv('vuelos/Cleaned_dataset.csv')
print(df.head())

df.isnull().sum()

# Una información rápida sobre los datos
print(df.info())

# Descripción estadística de los datos
print(df.describe())

#df.corr().T

# Tamaño de los datos 
print(df.shape)

df = df.dropna()
df.drop_duplicates( keep=False, inplace=True)
df = df.reset_index(drop = True)
print(df.shape)

df1=df.groupby(['Airline','Flight_code'],as_index=False).count()
df1.Airline.value_counts()

#Indigo se convierte en la aerolínea más popular
df2 = df.groupby(['Flight_code','Airline','Class'],as_index=False).count()
df2['Class'].value_counts()

#La mayoría de las aerolíneas tienen clase económica como común
plt.figure(figsize=(15,15))
df2['Class'].value_counts().plot(kind='pie',textprops={'color':'black'},autopct='%.2f',cmap='cool')
plt.title('Clases de diferentes aerolíneas\n',fontsize=16, fontweight = "bold")
plt.legend(['Económica','Negocio'])
plt.show()

#¿Los precios varían con las Aerolíneas?

#Como podemos ver Vistara tiene rango de precio máximo.
#Vistara y Air_India Airlines tienen precio máximo en comparación con otras
#SpiceJet, AirAsia, GO_First e Indigo tienen precios similares
fig = px.box(df, y = "Fare", x = 'Airline', color_discrete_sequence = ["orange"], template = "plotly_dark")
fig.show()

#¿El precio del billete varía entre clase económica y clase ejecutiva?

#El precio del billete es máximo para la clase Business en comparación con la clase Economy
plt.figure(figsize=(15,15))
sns.boxplot(x='Class',y='Fare',data=df,palette='hls')
plt.title('Clase Vs Precio Del Boleto\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Clase\n',fontsize=12)
plt.ylabel('Precio\n',fontsize=12)
plt.show()

#¿Cómo varía el precio del billete según el número de escalas de un vuelo?

#Los vuelos que tienen una escala tienen precio máximo de boleto
print(df.columns)

plt.figure(figsize=(15,15))
sns.boxplot(x='Total_stops',y='Fare',data=df,palette='hls')
plt.title('Paradas versus precio del boleto\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Paradas\n',fontsize=12)
plt.ylabel('Precio\n',fontsize=12)
plt.show()

"""¿Cómo cambia el precio del billete según la hora de salida y la hora de llegada?

1. Hora de salida versus precio del boleto
El precio del billete es mayor para los vuelos cuando la hora de salida es de noche
El precio del billete es casi igual para vuelos con horario de salida temprano en la mañana, mañana y tarde.
El precio del billete es bajo para los vuelos que tienen hora de salida tarde en la noche.

2. Hora de llegada versus precio del boleto
El precio del billete es mayor para los vuelos cuando la hora de llegada es por la tarde.
El precio del billete es casi igual para los vuelos. La hora de llegada es por la mañana y por la noche.
El precio del billete es bajo para los vuelos que tienen una hora de llegada tarde en la noche igual que la hora de salida"""
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
sns.boxplot(x='Departure',y='Fare',data=df)
plt.title('Hora de salida y precio del billete\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Hora de salida\n',fontsize=12)
plt.ylabel('Precio\n',fontsize=12)
plt.style.use('dark_background')
plt.subplot(1,2,2)
sns.boxplot(x='Arrival',y='Fare',data=df,palette='hls')
plt.title('Hora de llegada y precio del billete\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Hora de llegada\n',fontsize=12)
plt.ylabel('Precio\n',fontsize=12)
plt.show()

"""¿Cómo cambia el precio con el cambio en la ciudad de origen y la ciudad de destino?

1. Ciudad de origen versus precio de la entrada
El precio del billete es mayor para los vuelos cuya ciudad de origen es Calcuta
El precio del billete es casi igual para vuelos que tienen ciudades de origen como Mumbai y Chennai, Hyderabad y Bangalore.
El precio del billete es bajo para los vuelos que tienen como ciudad de origen Delhi.

2. Ciudad de destino versus precio del boleto
El precio del billete es mayor para los vuelos cuya ciudad de destino es Calcuta y Chennai.
El precio del billete es casi igual para vuelos que tienen ciudades de destino como Mumbai y Bangalore.
El precio del billete es bajo para los vuelos que tienen como ciudad de destino Delhi."""

print(df.columns)
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
sns.boxplot(x='Source',y='Fare',data=df)
plt.title('Ciudad de origen versus precio de la entrada\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Ciudad de origen\n',fontsize=12)
plt.ylabel('Precio\n',fontsize=12)
plt.subplot(1,2,2)
sns.boxplot(x='Destination',y='Fare',data=df,palette='hls')
plt.title('Ciudad de destino versus precio del boleto\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Ciudad de destino\n',fontsize=12)
plt.ylabel('Precio\n',fontsize=12)
plt.show()

#¿Cómo varía el precio con la duración del vuelo según la clase?

#Con el aumento de la duración, el precio del billete también aumenta tanto en la clase económica como en la clase ejecutiva
plt.figure(figsize=(15,15))
sns.lineplot(data = df,x = 'Duration_in_hours',y = 'Fare',hue = 'Class',palette = 'hls')
plt.title('Precio del billete versus duración del vuelo según la clase\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Duración\n',fontsize = 12)
plt.ylabel('Precio\n',fontsize = 12)
plt.show()

#¿Cómo afecta el precio en los días que faltan para la Salida?

#Como podemos ver en comparación con otros, cuando quedan dos días para la salida, el precio del billete es muy alto para todas las aerolíneas.
plt.figure(figsize=(15,15))
sns.lineplot(data=df,x='Days_left',y='Fare',color='blue')
plt.title('Días restantes para la salida versus precio del boleto\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Días restantes para la salida\n',fontsize=12)
plt.ylabel('Precio\n',fontsize=12)
plt.show()

plt.figure(figsize=(15,15))
sns.lineplot(data=df,x='Days_left',y='Fare',color='blue',hue='Airline',palette='hls', marker = "v")
plt.title('Días restantes para la salida versus precio del boleto de cada aerolínea\n', fontsize = '16', fontweight = 'bold')
plt.legend(fontsize=12)
plt.xlabel('Días restantes para la salida\n',fontsize=12)
plt.ylabel('Precio\n',fontsize=12)
plt.show()

#Número total de Vuelos de una ciudad a otra
df.groupby(['Flight_code','Source','Destination','Airline','Class'],as_index=False).count().groupby(['Source','Destination'],as_index=False)['Flight_code'].count().head(10)

#Precio promedio de diferentes aerolíneas desde la ciudad de origen hasta la ciudad de destino
df.groupby(['Airline','Source','Destination'],as_index=False)['Fare'].mean().head(10)

# Convertir las etiquetas en forma numérica usando Label Encoder
le=LabelEncoder()
for col in df.columns:
    if df[col].dtype=='object':
        df[col]=le.fit_transform(df[col])
# almacenar las variables dependientes en X y la variable independiente en Y
x=df.drop(['Fare'],axis=1)
y=df['Fare']

# Dividir los datos en conjunto de entrenamiento y conjunto de prueba
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=42)
x_train.shape,x_test.shape,y_train.shape,y_test.shape

# Escalar los valores para convertir los valores int a lenguajes de máquina
mmscaler=MinMaxScaler(feature_range=(0,1))
x_train=mmscaler.fit_transform(x_train)
x_test=mmscaler.fit_transform(x_test)
x_train=pd.DataFrame(x_train)
x_test=pd.DataFrame(x_test)  
a={'Model Name':[], 'Mean_Absolute_Error_MAE':[] ,'Adj_R_Square':[] ,'Root_Mean_Squared_Error_RMSE':[] ,'Mean_Absolute_Percentage_Error_MAPE':[] ,'Mean_Squared_Error_MSE':[] ,'Root_Mean_Squared_Log_Error_RMSLE':[] ,'R2_score':[]}
Results=pd.DataFrame(a)
Results.head()

# Cree objetos de modelos de regresión/regresor con hiperparámetros predeterminados
modelmlg = LinearRegression()
modeldcr = DecisionTreeRegressor()
#modelbag = BaggingRegressor()
modelrfr = RandomForestRegressor()
modelSVR = SVR()
modelXGR = xgb.XGBRegressor()
modelKNN = KNeighborsRegressor(n_neighbors=5)
modelETR = ExtraTreesRegressor()
modelRE=Ridge()
modelLO=linear_model.Lasso(alpha=0.1)

modelGBR = GradientBoostingRegressor(loss='squared_error', learning_rate=0.1, n_estimators=100, subsample=1.0,
                                     criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1,
                                     min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0,
                                     init=None, random_state=None, max_features=None,
                                     alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False,
                                     validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0)

# Matriz de evaluación de todos los algoritmos
MM = [modelmlg, modeldcr, modelETR, modelGBR, modelXGR, modelRE, modelLO]

for models in tqdm_notebook(MM):
    
    # Ajustar el modelo con datos del tren
    
    models.fit(x_train, y_train)
    
    # Predecir el modelo con datos de prueba

    y_pred = models.predict(x_test)
    
    # Imprimir el nombre del modelo
    
    #print('Model Name: ', models)
    
    # Métricas de evaluación para el análisis de regresión

    # print('Error absoluto medio (MAE):', round(metrics.mean_absolute_error(y_test, y_pred),3))  
    # print('Error cuadrático medio (MSE):', round(metrics.mean_squared_error(y_test, y_pred),3))  
    # print('Error cuadrático medio (RMSE):', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),3))
    # print('R2_puntuación:', round(metrics.r2_score(y_test, y_pred),6))
    # print('Error de registro cuadrático medio (RMSLE):', round(np.log(np.sqrt(metrics.mean_squared_error(y_test, y_pred))),3))
    
    # Defina la función para calcular el MAPE - Error porcentual absoluto medio
    def MAPE (y_test, y_pred):
        y_test, y_pred = np.array(y_test), np.array(y_pred)
        return np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    # Evaluación de MAPE
    result = MAPE(y_test, y_pred)
    # print('Error porcentual absoluto medio (MAPE):', round(result, 2), '%')
    
    # Calcular valores de R cuadrado ajustados
    r_squared = round(metrics.r2_score(y_test, y_pred),6)
    adjusted_r_squared = round(1 - (1-r_squared)*(len(y)-1)/(len(y)-x.shape[1]-1),6)
    # print('Adj R Cuadrado: ', ajustado_r_cuadrado)
    # print('------------------------------------------------------------------------------------------------------------')
    #-------------------------------------------------------------------------------------------
    new_row = {'Model Name' : models,
               'Mean_Absolute_Error_MAE' : metrics.mean_absolute_error(y_test, y_pred),
               'Adj_R_Square' : adjusted_r_squared,
               'Root_Mean_Squared_Error_RMSE' : np.sqrt(metrics.mean_squared_error(y_test, y_pred)),
               'Mean_Absolute_Percentage_Error_MAPE' : result,
               'Mean_Squared_Error_MSE' : metrics.mean_squared_error(y_test, y_pred),
               'Root_Mean_Squared_Log_Error_RMSLE': np.log(np.sqrt(metrics.mean_squared_error(y_test, y_pred))),
               'R2_score' : metrics.r2_score(y_test, y_pred)}
    #Results = Results.append(new_row, ignore_index=True)
    #------------------------------------------------------------
    
#print(Results)

models=['LinearRegression','DecisionTreeRegressor','RandomForestRegressor','KNeighborsRegressor','ExtraTreesRegressor','GradientBoostingRegressor','XGBRegressor']
result=pd.DataFrame({'Model_Name':models})
result['Adj_R_Square']=Results['Adj_R_Square']
result['Mean_Absolute_Error_MAE']=Results['Mean_Absolute_Error_MAE']
result['Root_Mean_Squared_Error_RMSE']=Results['Root_Mean_Squared_Error_RMSE']
result['Mean_Absolute_Percentage_Error_MAPE']=Results['Mean_Absolute_Percentage_Error_MAPE']
result['Mean_Squared_Error_MSE']=Results['Mean_Squared_Error_MSE']
result['Root_Mean_Squared_Log_Error_RMSLE']=Results['Root_Mean_Squared_Log_Error_RMSLE']
result['R2_score']=Results['R2_score']
result=result.sort_values(by='Adj_R_Square',ascending=False).reset_index(drop=True)
print(result)

#De los resultados anteriores, los 3 modelos principales al comparar errores, los valores Adj_R_Square y R2_Score son
#1. ExtraTreesRegressor 2. DecisionTreeRegressor 3. KNeighborsRegressor
#Entrenando los datos con ExtraTreesRegressor

#Entrenando al modelo con
modelETR.fit(x_train, y_train)
    
# Predecir el modelo con datos de prueba
y_pred = modelETR.predict(x_test)
out=pd.DataFrame({'Price_actual':y_test,'Price_pred':y_pred})
result=df.merge(out,left_index=True,right_index=True)
result.sample(10)
plt.figure(figsize=(20,8))
sns.lineplot(data=result,x='Days_left',y='Price_actual',color='fuchsia')
sns.lineplot(data=result,x='Days_left',y='Price_pred',color='cyan')
plt.title('Días restantes para la salida versus el precio real del boleto y el precio previsto del boleto\n', fontsize = '16', fontweight = 'bold')
plt.legend(labels=['Precio real','Predicción de precios'],fontsize=12)
plt.xlabel('Días restantes para la salida\n',fontsize=12)
plt.ylabel('Precio real y previsto\n',fontsize=12)
plt.show()

plt.figure(figsize=(10,5))
sns.regplot(x='Price_actual',y='Price_pred',data=result,color='cyan')
plt.title('Precio real versus precio previsto\n ', fontsize = '16', fontweight = 'bold')
plt.xlabel('Precio actual\n',fontsize=12)
plt.ylabel('Precio previsto\n',fontsize=12)
plt.show()

"""Introducción
El conjunto de datos Airfare ML: Predicting Flight Fares es una colección de precios de vuelos y características relacionadas para varias rutas 
entre diferentes ciudades. El conjunto de datos está disponible en Kaggle y fue compilado por Yash Dharme. El propósito de este conjunto de datos 
es proporcionar un recurso útil para crear modelos de aprendizaje automático que puedan predecir el precio de los vuelos entre diferentes ciudades.

El conjunto de datos contiene más de 10.000 registros, cada uno de los cuales representa una ruta de vuelo única. Las características 
proporcionadas incluyen la aerolínea, los aeropuertos de origen y destino, la duración del vuelo, la distancia entre los aeropuertos, el número 
de escalas y varios otros atributos. La variable objetivo es el precio del vuelo, que se proporciona en rupias indias (INR).

"""

# Configurar una semilla aleatoria
np.random.seed(42)

# Carga del conjunto de datos
#Cargar los datos son los primeros pasos hacia la creación de un modelo de aprendizaje automático. Hagamoslo.

# Especificar ruta raíz
data_path = 'vuelos/Cleaned_dataset.csv'

# Leer el archivo CSV
data_frame = pd.read_csv(data_path)

# Hacer una copia
org_df = data_frame.copy()

# Vistazo rápido
data_frame.head()
"""
El conjunto de datos considerado contiene una variedad de tipos de datos, incluidas fechas, valores categóricos y valores numéricos. Para analizar 
los datos de forma eficaz, es importante procesar y manejar adecuadamente cada uno de estos tipos de datos.

Vale la pena señalar que se ha creado una copia del marco de datos original para facilitar la visualización efectiva de los datos. Al crear una 
copia separada de los datos, podemos preservar la integridad de los datos originales y garantizar que cualquier modificación realizada con fines 
de visualización no afecte los pasos posteriores del procesamiento de datos.

Para llevar a cabo el procesamiento previo de datos, utilizaremos un marco de datos independiente que se ha creado específicamente para este fin. 
Este enfoque nos permite mantener intactos los datos originales e implementar los cambios necesarios de manera controlada, sin riesgo de pérdida 
o corrupción de datos importantes.
"""

org_df.isnull().sum()

"""
No faltan valores en el conjunto de datos, lo que significa que podemos continuar con la visualización de datos sin problemas 
ni preocupaciones relacionadas con datos incompletos.

Visualización de datos
Explorar y analizar datos es un paso esencial en cualquier proyecto de ciencia de datos, ya que ayuda a obtener una comprensión más profunda de 
los patrones y relaciones subyacentes en los datos. Para realizar este análisis de forma eficaz, utilizaremos potentes herramientas de 
visualización de datos como Matplotlib, Seaborn y Plotly. Comencemos explorando la relación entre las columnas de características y la columna de 
destino.

veamos qué día de la semana es el más popular para un viaje.
"""
# Calcular distribución
journey_day_counts = org_df.Journey_day.value_counts()
journey_day_names  = journey_day_counts.index

# Distribución de parcelas
fig = px.pie(names=journey_day_names, values=journey_day_counts, hole=0.2, title="Distribución del día de viaje\n", template = "plotly_dark")
fig.show()

fig = px.bar(x=journey_day_names, y=journey_day_counts, color=journey_day_names, template = "plotly_dark")
fig.show()

#Aunque el promedio es más o menos el mismo a lo largo de los días, hay una diferencia notable en el volumen de tráfico: el lunes muestra un 
# volumen mucho mayor en comparación con los otros días. De hecho, el lunes tiene el mayor volumen de tráfico mientras que el domingo tiene el 
# menor. Además, hay un patrón ligeramente decreciente en el volumen de tráfico de lunes a domingo. Estos conocimientos sugieren que el volumen 
# de tráfico varía significativamente entre los diferentes días de la semana, siendo el lunes el día más ocupado y el domingo el menos ocupado.

fig = px.histogram(org_df, x='Journey_day', color='Airline', text_auto=True, barmode='stack', title="Día de viaje versus aerolínea\n", template = "plotly_dark")
fig.show()

#Este gráfico categórico muestra claramente que la distribución de las aerolíneas que operan es relativamente consistente en todos los días. 
# Independientemente del día de la semana, el número de vuelos operados por cada aerolínea es aproximadamente el mismo. Esto indica que no hay 
# una variación significativa en los niveles de actividad de la industria aérea según el día de la semana.

#La baja frecuencia de algunas aerolíneas en el conjunto de datos indica que están menos representadas y pueden tener menos influencia en la 
# variable objetivo que se está estudiando. Sin embargo, es importante señalar que el impacto de una aerolínea en la variable objetivo no puede 
# determinarse únicamente por su frecuencia en el conjunto de datos, y se requieren análisis más profundos para comprender completamente su 
# impacto.

fig = px.histogram(org_df, x='Journey_day', color='Class', text_auto=True, barmode='stack', title="Día de viaje versus clase\n", template = "plotly_dark")
fig.show()

#Según los datos, parece que los pasajeros no tienen una fuerte preferencia por una clase en particular cuando eligen viajar en un determinado 
# día de la semana. La distribución de clases (económica, económica premium, business y primera) parece ser más o menos la misma en todos los 
# días de la semana. Esto sugiere que los pasajeros pueden elegir su clase en función de otros factores.

#También podemos observar que los pasajeros de primera clase son extremadamente raros.

fig = px.histogram(org_df, x='Journey_day', color='Departure', text_auto=True, barmode='stack', title="Día de viaje versus salida\n", template = "plotly_dark")
fig.show()

#Es importante tener en cuenta que el día de la semana no parece tener un impacto significativo en el momento en que las personas eligen tomar 
# sus vuelos. Los horarios de salida para todos los días de la semana son relativamente consistentes.

fig = px.box(org_df, x='Journey_day', y='Fare', color='Journey_day', title="Día de viaje versus tarifa\n", template = "plotly_dark")
fig.show()

fig = px.violin(org_df, x='Journey_day', y='Fare', color='Journey_day', template = "plotly_dark")
fig.show()

#Lo primero que nos llama inmediatamente la atención son los valores atípicos presentes en los datos. El diagrama de caja está lleno de estos 
# puntos atípicos, lo que sugiere que tendremos que tratarlos adecuadamente durante la etapa de preprocesamiento de datos.

#Al analizar la distribución de los precios justos con respecto al día de viaje, podemos ver que es relativamente consistente en todos los días 
# de la semana. Esto implica que los precios justos no se ven afectados significativamente por el día del viaje. Sin embargo, vale la pena 
# señalar que si bien el valor mediano se mantiene constante a lo largo de los días, los valores máximos, así como sus distribuciones, sí 
# muestran cierta variación.

# Calcular distribución
airline_counts = org_df.Airline.value_counts()
airline_names  = airline_counts.index

# Distribución de parcelas
fig = px.pie(names=airline_names, values=airline_counts, hole=0.2, title="Distribución de aerolíneas\n", template = "plotly_dark")
fig.show()

fig = px.bar(x=airline_names, y=airline_counts, color=airline_names, template = "plotly_dark")
fig.show()

#Al analizar el gráfico circular y el gráfico de barras, es evidente que Vistara y Air India son los líderes del mercado, Vistara posee la mayor 
# parte de la participación de mercado, representando el 50% de la participación total del mercado, mientras que Air India ocupa el segundo lugar 
# con 26% de la cuota de mercado. Esto muestra claramente su fuerte dominio en el mercado de las aerolíneas e indica su éxito en capturar una 
# porción significativa de la base de clientes.

#Después de Vistara y Air India, Indigo ocupa el tercer lugar entre los líderes del mercado con una participación de mercado del 16%. Aunque su 
# cuota de mercado es inferior a la de las dos principales aerolíneas, todavía tiene una cuota de mercado significativa. Vale la pena señalar que 
# la brecha entre Indigo y Air India no es demasiado grande, ya que Air India tiene una participación de mercado del 26%. Esto sugiere que Indigo 
# es un fuerte competidor en el mercado, a pesar de no tener el mismo nivel de dominio que Vistara y Air India.

fig = px.histogram(org_df, x='Airline', color='Class', text_auto=True, barmode='stack', title="Aerolínea vs Clase\n", template = "plotly_dark")
fig.show()

#Los datos reafirman nuestro entendimiento inicial de que Vistara ofrece la gama más completa de opciones de vuelo, incluidos vuelos económicos, 
# económicos premium, ejecutivos y de primera clase, mientras que Air India solo ofrece vuelos económicos, ejecutivos y de primera clase, y todas 
# las demás aerolíneas solo ofrecen la clase económica. Esto resalta claramente el hecho de que la gama de opciones de vuelo ofrecidas por una 
# aerolínea puede afectar significativamente su estrategia de precios, y los precios de Vistara y Air India reflejan sus respectivas ofertas.

fig = px.box(org_df, x='Airline', y='Fare', color='Airline', title="Aerolínea versus tarifa\n", template = "plotly_dark")
fig.show()

fig = px.violin(org_df, x='Airline', y='Fare', color='Airline', template = "plotly_dark")
fig.show()

#Como era de esperar, Vistara y Air India, los líderes del mercado con mayor participación de mercado, también tienen los precios más altos. Esto 
# podría atribuirse a su oferta de vuelos de negocios, económicos y económicos premium, que ofrecen mayor comodidad y lujo a los pasajeros.

#Por otro lado, otras aerolíneas solo ofrecen vuelos en clase económica, lo que resulta en una distribución similar de precios entre ellas. A 
# pesar de la diferencia en las clases ofrecidas, los valores medios de todas las aerolíneas son aproximadamente los mismos. Por tanto, podemos 
# concluir que la clase de vuelos que ofrece una aerolínea tiene un impacto significativo en su estrategia de precios.

# Calcular la distribución de clases
class_dis = org_df.Class.value_counts()
class_dis_names = class_dis.index

# Mostrar gráfico circular
fig = px.pie(names=class_dis_names, values=class_dis, hole=0.4, title="Distribución de datos de clase\n", template = "plotly_dark")
fig.show()

fig = px.bar(x=class_dis_names, y=class_dis, color=class_dis_names, template = "plotly_dark")
fig.show()

#Esto era totalmente esperado. Si observamos el gráfico circular y el gráfico de barras, podemos ver claramente que los pasajeros de clase 
# económica cubren casi el 56% del total de pasajeros. Seguido por los pasajeros de clase ejecutiva con un 28% de participación. Y después de la 
# clase ejecutiva, está la clase económica premium, con una proporción de 16. Donde los pasajeros de primera clase son extremadamente bajos. Es 
# por eso que personalmente considero esto como un valor atípico del conjunto de datos normal, porque realmente no pertenece a la distribución de 
# datos normal. El recuento total en porcentaje es inferior al 0,04%.
fig = px.box(org_df, x='Class', y='Fare', color='Class', template = "plotly_dark")
fig.show()

fig = px.violin(org_df, x='Class', y='Fare', color='Class', template = "plotly_dark")
fig.show()

"""
La relación entre la clase de vuelo y el precio del vuelo es interesante y los datos revelan algunas ideas intrigantes. A pesar de que la clase 
económica es la clase más popular con una participación del 56% del total de pasajeros, su precio es el menor entre todas las clases. 
Probablemente este sea un resultado esperado.

Por otro lado, la clase económica premium tiene un rango de precios ligeramente más alto, mientras que la clase ejecutiva tiene los precios más 
altos. Sin embargo, cuando observamos el diagrama de caja, también podemos identificar algunos valores atípicos, que indican que algunos vuelos 
tienen tarifas extremadamente altas que requieren más investigación.

Por último, la primera clase tiene una distribución similar a la de la clase ejecutiva, pero sus precios de tarifas se limitan a un rango muy 
corto y su frecuencia es extremadamente baja..
"""

stop_dis = org_df.Total_stops.value_counts()
stop_dis_names = stop_dis.index

# Gráfico circular
fig = px.pie(names=stop_dis_names, values=stop_dis, hole=0.4, title="Detiene la distribución\n", template = "plotly_dark")
fig.show()

# Gráfico de barras
fig = px.bar(x=stop_dis_names, y=stop_dis, color=stop_dis_names, template = "plotly_dark")
fig.show()

#Al analizar los datos, hemos descubierto que apenas el 14% del total de vuelos de nuestro estudio se clasifican como directos, es decir, que no 
# tienen escalas intermedias entre el aeropuerto de origen y destino. La mayoría de los vuelos, que representan aproximadamente el 81%, tienen al 
# menos una escala a lo largo de la ruta, lo que indica que las escalas son algo común en los viajes aéreos. Curiosamente, sólo una pequeña 
# proporción de los vuelos, apenas el 6%, tienen dos o más escalas intermedias, lo que sugiere que los vuelos con escalas múltiples son 
# relativamente poco comunes en la industria aérea.

fig = px.box(org_df, x='Total_stops', y='Fare', color='Total_stops', template = "plotly_dark")
fig.show()

fig = px.violin(org_df, x='Total_stops', y='Fare', color='Total_stops', template = "plotly_dark")
fig.show()

"""
Nuestro análisis ha revelado que los vuelos sin escalas son el tipo de vuelo que se ofrece con más frecuencia y, como resultado, tienden a tener 
un rango de precios más alto en comparación con los vuelos con una o dos escalas. El diagrama de caja muestra que el rango de precios de los 
vuelos sin escalas es significativamente más amplio, lo que indica que los precios de los boletos para vuelos sin escalas varían mucho.

Curiosamente, el precio medio de los vuelos sin escalas es comparable al de los vuelos con dos o más escalas, a pesar de que la distribución 
vertical de los vuelos sin escalas es mayor en el diagrama de caja. Esto sugiere que, si bien los vuelos sin escalas pueden tener un rango de 
precios más amplio, el precio medio no es necesariamente más alto que el de los vuelos con múltiples escalas. Sin embargo, cabe señalar que la 
distribución de precios para vuelos con dos o más escalas se limita a un rango relativamente corto.
"""
fig = px.histogram(org_df, x='Fare', nbins=30, color_discrete_sequence=['coral'])
fig.update_layout(title='Histograma de tarifa\n', xaxis_title='Tarifa\n', yaxis_title='Conteo\n', 
                  bargap=0.1, bargroupgap=0.1, font=dict(size=12), template = "plotly_dark")
fig.update_traces(marker=dict(line=dict(width=1, color='black')))
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='black')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='black')
fig.update_layout(margin=dict(l=50, r=50, t=50, b=50))
fig.show()

# Cree un gráfico de contorno de densidad de la columna Tarifa
fig = px.density_contour(org_df, x='Fare',  
                          marginal_x='histogram', marginal_y='violin', template = "plotly_dark")
fig.update_layout(title='Gráfico de contorno de densidad de tarifa\n',
                  xaxis_title='Tarifa\n',
                  yaxis_title='Densidad\n')
fig.show()

# Cree un gráfico de mapa de calor de densidad de la columna Tarifa
fig = px.density_heatmap(org_df, x='Fare', nbinsx=20, nbinsy=20, 
                          color_continuous_scale='Magma', template = "plotly_dark")
fig.update_layout(title='Mapa de calor de densidad de la tarifa\n',
                  xaxis_title='Tarifa\n',
                  yaxis_title='Densidad\n')
fig.show()

"""
Al observar los gráficos de histograma y densidad, podemos ver que la distribución de los precios de las tarifas está sesgada hacia la derecha, 
con la mayoría de los valores entre 0 y 20.000. Sin embargo, hay un pequeño pico alrededor de 40 000-60 000, lo que indica que hay algunas tarifas
más caras en el conjunto de datos.

El gráfico de densidad también muestra que hay algunos valores de tarifas extremadamente altos después de 70K, que pueden considerarse valores 
atípicos. Si bien puede haber algunos casos especiales en los que estas tarifas elevadas sean legítimas, su recuento es extremadamente bajo en 
comparación con la distribución normal de datos.

Preprocesamiento de datos
Como hemos identificado los posibles valores atípicos en nuestros datos, ahora podemos preprocesar los datos para filtrar estos valores atípicos 
y aplicar otras transformaciones necesarias. Este paso es crucial para garantizar que nuestro modelo no se vea afectado por valores extremos, que 
pueden afectar significativamente su rendimiento.
"""

# Eliminar 'AkasaAir', 'AllianceAir' y 'StarAir' de los datos
data_frame = data_frame[~data_frame['Airline'].isin(['AkasaAir', 'AllianceAir', 'StarAir'])]

# Confirmación visual
data_frame.Airline.value_counts()

#La razón por la que eliminamos las filas que contienen los valores de 3 aerolíneas (AkasaAir, AllianceAir y StarAir) es que estas aerolíneas 
# tienen un recuento de frecuencia significativamente bajo en comparación con las otras aerolíneas en el conjunto de datos. Esto indica que los 
# datos de estas aerolíneas no son suficientes para hacer una generalización o conclusión sobre el precio de la tarifa.

#Además, luego de analizar los datos, observamos que las tarifas de estas aerolíneas no tienen un impacto significativo en la columna de tarifas 
# objetivo. Eliminar estas filas del conjunto de datos ayuda a mejorar la precisión y confiabilidad de nuestro análisis al reducir la posibilidad 
# de valores atípicos y resultados erróneos.

# Eliminar la clase 'Primera'
data_frame = data_frame[~(data_frame['Class'] == 'First')]

# Confirmación visual
data_frame.Class.value_counts()

#De manera similar a la columna 'Aerolínea', hemos eliminado los valores 'Primeros' de la columna 'Clase' debido a su baja frecuencia en 
# comparación con las otras clases. Además, según nuestro análisis, encontramos que las tarifas de 'Primera' clase no tienen un impacto 
# significativo en la variable objetivo en comparación con las otras clases. Por lo tanto, eliminar estas filas no afectará nuestro análisis y 
# mejorará la precisión de nuestros resultados.

# Eliminar valores de 'Tarifa' > 70K
data_frame = data_frame[~(data_frame['Fare'] > 70000)]

# Confirmación visual
data_frame.Fare.describe()

"""
Según nuestro análisis de datos, eliminamos los valores en los que la tarifa era superior a 70 000. Hubo múltiples razones para hacerlo.

En primer lugar, el recuento de estos valores fue extremadamente bajo, lo que significa que no eran representativos de la mayoría de los datos.

En segundo lugar, al analizar el histograma y el gráfico de densidad de la columna de tarifas, estos valores parecían ser valores atípicos y no 
caían dentro de la distribución normal de datos.
"""

# Dividir datos en columnas de características y de destino
X = data_frame.drop('Fare', axis=1)
Y = data_frame['Fare']
print(X.head())

#Eliminemos algunas columnas no deseadas, como la fecha del viaje y el código de vuelo.

# Eliminar columnas no deseadas
X.drop(columns=['Date_of_journey', 'Flight_code'], inplace=True)

# Vistazo rápido
print(X.head())

#Ahora que hemos eliminado las columnas y valores no deseados, debemos centrarnos en tratar los datos categóricos. Como los algoritmos de 
# aprendizaje automático suelen trabajar con datos numéricos, necesitamos convertir los datos categóricos a forma numérica. Esto se puede hacer 
# usando un mapeo de diccionario simple, donde asignamos un valor entero único a cada categoría, o usando la función LabelEncoder de scikit-learn.

# Columnas categóricas 
cat_cols = [col for col in X.columns if X[col].dtype=='O']
print(f"Categorical Columns : {cat_cols}")

# Aplicar el codificador de etiquetas
X[cat_cols] = X[cat_cols].apply(lambda x: LabelEncoder().fit_transform(x))
X.head()

#Debido a la baja presencia de columnas numéricas, solo tenemos "duración en horas", que tienen valores muy altos en comparación con las otras 
# columnas. Por lo tanto, será mejor utilizar la escala estándar en los datos para que estén a la misma escala que las otras funciones.

# Inicializar escalador
scaler = StandardScaler()

# Aplicar
X_scaled = scaler.fit_transform(X)
X_scaled # matriz numerosa

"""
Ahora que hemos preprocesado nuestros datos, pasemos a uno de los pasos más importantes: la selección de funciones.

Selección de características
La selección de funciones es un paso crucial en el aprendizaje automático, ya que ayuda a identificar las funciones o variables más relevantes 
que contribuyen a la variable objetivo. Este proceso se realiza para mejorar la precisión y eficiencia del modelo, eliminando características 
irrelevantes o redundantes que podrían causar un sobreajuste o agregar ruido al modelo.
"""

# Correlación de Pearson
X['Fare'] = Y
corr = X.corr()
corr = np.round(corr, 2)

# Visualización de mapas de calor
fig = px.imshow(corr, text_auto=True, height=800, template = "plotly_dark")
fig.update_layout(
    title={
        'text': "Mapa de calor de correlación de Pearson",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    xaxis_title="Características\n",
    yaxis_title="Características\n",
)
fig.show()
X.drop(columns=['Fare'], inplace=True)

"""
Al observar la correlación, podemos afirmar que no existen fuertes relaciones positivas ni fuertes relaciones negativas entre las características.
Sin embargo, podemos encontrar una fuerte relación negativa entre la columna de destino Tarifa y la columna de característica 'clase'.

Aparte de eso, existen relaciones pequeñas, pero positivas y negativas, entre la columna de destino y las columnas de características, es decir, 
paradas totales y duración en horas.

Si observamos las columnas de características, solo hay una fuerte correlación negativa entre la duración en horas y el total de paradas.
"""

# Crea una nueva instancia del regresor de bosque aleatorio
rfr = RandomForestRegressor()

# Entrene el modelo en la matriz de características escalada (X_scaled) y la variable objetivo (Y)
rfr.fit(X_scaled, Y)

# Recopile las importancias de las características para cada característica de entrada
feature_importances = rfr.feature_importances_

# Imprimir las características importantes
print("Importancia de las características:")
for feature, importance in zip(X.columns, feature_importances):
    print("{:20}: {:.3f}".format(feature, importance))
    
# Crear un gráfico circular de importancia de las características
fig_pie = px.pie(names=X.columns, values=feature_importances, color=X.columns, hole=0.4, title="Importancia de las funciones (gráfico circular)\n", template = "plotly_dark")
fig_pie.update_traces(textposition='inside', textinfo='percent+label')
fig_pie.update_layout(showlegend=False)
fig_pie.show()

# Crear un gráfico de barras de características importantes
fig_bar = px.bar(x=X.columns, y=feature_importances, color=X.columns, title="Importancia de las funciones (gráfico de barras)\n", template = "plotly_dark")
fig_bar.update_layout(xaxis_title="Características\n", yaxis_title="Importancia\n", showlegend=False)
fig_bar.show()

"""
Después de analizar tanto la correlación como la importancia de las características del modelo de bosque aleatorio, queda claro que la columna 
"Clase" tiene un impacto significativo en la columna "Tarifa". Esta columna parece ser la única característica que tiene una correlación directa 
con la variable objetivo. Mientras que otras columnas tienen algún efecto, es insignificante.

Según estos hallazgos, sería prudente mantener las tres características principales que tienen una correlación directa con la columna "Clase". 
Esto nos permitirá crear un conjunto de funciones más centrado que priorice las funciones más importantes.

Sin embargo, dada la abrumadora importancia de la columna "Clase", también puede valer la pena crear otro conjunto de características que incluya 
solo la columna "Clase" como vector de características. Esto nos permitirá explorar la relación entre la columna "Clase" y la columna "Tarifa" 
con mayor detalle, sin la influencia confusa de otras características.

SelectKBest es un método de selección de funciones en Scikit-learn que selecciona las k mejores funciones en función de una función de puntuación 
específica. Es una técnica de selección de características supervisada que aprovecha la relación entre las características y la variable objetivo. 
La función de puntuación se utiliza para evaluar la importancia de cada característica. Las funciones de puntuación más utilizadas son chi-cuadrado, 
información mutua y puntuación f.

La regresión F es un método de selección de características que utiliza una prueba F para evaluar la importancia de cada característica en la 
predicción de la variable objetivo.

La regresión F funciona calculando la estadística F y el valor p correspondiente para cada característica. El estadístico F mide el grado en que 
la variación en la variable objetivo puede explicarse por la variación en la característica, en relación con la variación residual.

El valor p mide la importancia del estadístico F e indica la probabilidad de obtener un estadístico F tan extremo o más extremo que el valor 
observado, bajo la hipótesis nula de que la característica no tiene efecto sobre la variable objetivo.

La información mutua es una medida de la dependencia mutua entre dos variables. En el contexto de la selección de características, la información 
mutua mide la cantidad de información que proporciona una característica sobre la variable objetivo. Es más probable que las características con 
alta información mutua sean informativas para predecir la variable objetivo.
"""

# Regresión F con SelectKBest
k_best = SelectKBest(f_regression, k=3)

# Ajustar el modelo a los datos
k_best.fit(X_scaled, Y)

# Obtenga la importancia de las funciones
feature_scores = k_best.scores_

# Mostrar puntuaciones
print("{:25} {:15}".format("Feature Names", "Score"))
for name, score in zip(X.columns, feature_scores):
    print("{:20}: {:10.3f}".format(name, score))
    
# Crear un gráfico circular de importancia de las características
fig_pie = px.pie(names=X.columns, values=feature_scores, color=X.columns, hole=0.4, title="Regresión F (gráfico circular)\n", template = "plotly_dark")
fig_pie.update_traces(textposition='inside', textinfo='percent+label')
fig_pie.update_layout(showlegend=False)
fig_pie.show()

# Crear un gráfico de barras de características importantes
fig_bar = px.bar(x=X.columns, y=feature_scores, color=X.columns, title="Regresión F (gráfico de barras)\n", template = "plotly_dark")
fig_bar.update_layout(xaxis_title="Características\n", yaxis_title="Importancia\n", showlegend=False)
fig_bar.show()

# Regresión F con SelectKBest
k_best = SelectKBest(mutual_info_regression, k=3)

# Ajustar el modelo a los datos
k_best.fit(X_scaled, Y)

# Obtenga la importancia de las funciones
feature_scores = k_best.scores_

# Mostrar puntuaciones
print("{:25} {:15}".format("Feature Names", "Score"))
for name, score in zip(X.columns, feature_scores):
    print("{:20}: {:10.3f}".format(name, score))
    
# Crear un gráfico circular de importancia de las características
fig_pie = px.pie(names=X.columns, values=feature_scores, color=X.columns, hole=0.4, title="Información mutua (gráfico circular)\n", template = "plotly_dark")
fig_pie.update_traces(textposition='inside', textinfo='percent+label')
fig_pie.update_layout(showlegend=False)
fig_pie.show()

# Crear un gráfico de barras de características importantes
fig_bar = px.bar(x=X.columns, y=feature_scores, color=X.columns, title="Información mutua (gráfico de barras)\n", template = "plotly_dark")
fig_bar.update_layout(xaxis_title="Características\n", yaxis_title="Importancia\n", showlegend=False)
fig_bar.show()

"""
Si miramos la matriz de correlación, es evidente que "clase" y "duración en horas" son las dos características más importantes. Sin embargo, a 
medida que avanzamos, las cosas se vuelven más complejas. Utilizando el enfoque del método integrado para la selección de características con 
regresión forestal aleatoria, aún encontramos que la "clase" y la "duración en horas" son las dos características más importantes. 
Sorprendentemente, "quedan días" surgió como la tercera característica más importante.

Sin embargo, la puntuación de regresión F reveló una imagen ligeramente diferente en la que "clase", "duración en horas" y "paradas totales" se 
identificaron como las tres características principales. La relación de información mutua entre los vectores de características de regresión y el 
vector objetivo también reveló ideas interesantes. "Duración en horas" tuvo la mayor información mutua seguida de "destino", "fuente" y luego 
"clase".

Con base en esto, creemos un conjunto de datos con las cinco características principales. Estas cinco características son clase, duración en 
horas, día restante, paradas totales y fuente.
"""

# Recopile las 3 características más importantes
X_top_3 = X[['Class', 'Duration_in_hours', 'Days_left']]

# Recopile las 5 características más importantes
X_top_5 = X[['Class', 'Duration_in_hours', 'Days_left', 'Total_stops', 'Source']]

# Escale los datos de las 3 características principales
scaler = StandardScaler()
X_top_3 = scaler.fit_transform(X_top_3)

# Escale los datos de las 5 características principales
scaler = StandardScaler()
X_top_5 = scaler.fit_transform(X_top_5)
# Conjunto 1: datos completos
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X_scaled, Y, test_size=0.3, shuffle=True, random_state=42)

# Conjunto 2: 5 características principales
X_train_5, X_test_5, y_train_5, y_test_5 = train_test_split(X_top_5, Y, test_size=0.3, shuffle=True, random_state=42)

# Conjunto 3: Las 3 características principales
X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(X_top_3, Y, test_size=0.3, shuffle=True, random_state=42)

# Conjunto 4: característica más importante
X_train, X_test, y_train, y_test = train_test_split(X['Class'].to_numpy().reshape(-1,1), Y, test_size=0.3, shuffle=True, random_state=42)

#Para determinar el mejor ajuste del modelo para los tres conjuntos de datos, implementaremos un modelo de regresión lineal. Al utilizar un 
# modelo de regresión lineal, podemos identificar fácilmente la relación lineal entre las variables independientes y dependientes. Además, dado 
# que el tercer conjunto de datos contiene solo una característica, el enfoque más apropiado es utilizar un modelo de regresión lineal.

#Después de aplicar el modelo de regresión lineal en los tres conjuntos de datos, compararemos los resultados para identificar el conjunto de 
# datos que produzca los mejores resultados. Esta comparación nos ayudará a determinar el conjunto de datos que tiene la relación lineal más 
# significativa entre las variables independientes y dependientes. Al utilizar el conjunto de datos más adecuado, podemos hacer mejores 
# predicciones y decisiones para el dominio de nuestro problema.

def calculate_performance(y_true, y_pred):
    """
    Calcular las métricas de rendimiento para modelos de regresión.
    
    Parámetros:
    -----------
    y_true: forma similar a una matriz (n_samples,)
        Valores objetivo verdaderos.
    
    y_pred : forma similar a una matriz (n_samples,)
        Valores objetivo estimados.
    
    Devoluciones:
    --------
    lista
        Una lista de cuatro valores flotantes que representan el error cuadrático medio (RMSE), el error cuadrático medio (MSE),
        error absoluto medio (MAE) y puntuación R cuadrado (R^2), respectivamente.
    """
    # Calcular el error cuadrático medio (MSE)
    mse = mean_squared_error(y_true, y_pred)
    
    # Calcular el error absoluto medio (MAE)
    mae = mean_absolute_error(y_true, y_pred)
    
    # Calcular la puntuación R cuadrado (R^2)
    r2 = r2_score(y_true, y_pred)
    
    # Calcular el error cuadrático medio (RMSE)
    rmse = np.sqrt(mse)
    
    return [rmse, mse, mae, r2]
SCORE_NAMES = ['RMSE', 'MSE', 'MAE', 'R-sq']

# Entrenar y evaluar el modelo de regresión del árbol de decisión
dt_full = DTR()
dt_full.fit(X_train_full, y_train_full)
dt_full_pred = dt_full.predict(X_test_full)
dt_full_scores = calculate_performance(y_test_full, dt_full_pred)

# Imprimir puntuaciones de rendimiento
print("Puntuaciones de rendimiento de regresión del árbol de decisión:")
for name, score in zip(SCORE_NAMES, dt_full_scores):
    print("{:10} : {:.5f}".format(name, score))
    
# Regresión del árbol de decisión
dt_5 = DTR()

# Entrenar al modelo
dt_5.fit(X_train_5, y_train_5)

# Hacer predicciones
dt_5_pred = dt_5.predict(X_test_5)

# Evaluar el desempeño
dt_5_scores = calculate_performance(y_test_5, dt_5_pred)

# Mostrar métricas de rendimiento
print("Métricas de rendimiento de regresión del árbol de decisión (Subconjunto 5)")
print("----------------------------------------------------")
for name, score in zip(SCORE_NAMES, dt_5_scores):
    print("{:27} : {:.5f}".format(name, score))
    
# Regresión del árbol de decisión
dt_3 = DTR()
 
# Modelo de tren
dt_3.fit(X_train_3, y_train_3)

# Hacer predicción
dt_3_pred = dt_3.predict(X_test_3)

# Evaluar el desempeño
dt_3_scores = calculate_performance(y_test_3, dt_3_pred)

# Mostrar rendimiento
print("Métricas de rendimiento de regresión del árbol de decisión (Subconjunto 3)")
print("----------------------------------------------------")
for name, score in zip(SCORE_NAMES, dt_3_scores):
    print("{:10} : {:.5f}".format(name, score))
    
# Definir el modelo
dt = DTR()

# Entrenar el modelo
dt.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
dt_pred = dt.predict(X_test)

# Evaluar el rendimiento del modelo
dt_scores = calculate_performance(y_test, dt_pred)

# Imprimir las métricas de rendimiento
print("Métricas de rendimiento de regresión del árbol de decisión (Subconjunto 1)")
print("----------------------------------------------------")
for name, score in zip(SCORE_NAMES, dt_scores):
    print("{:<10}: {:.5f}".format(name, score))
    
DATA_SIZE = ['Full', 'Top-5', 'Top-3', 'Top-1']
RMSEs = [dt_full_scores[0], dt_5_scores[0], dt_3_scores[0], dt_scores[0]]
MSEs  = [dt_full_scores[1], dt_5_scores[1], dt_3_scores[1], dt_scores[1]]
MAEs  = [dt_full_scores[2], dt_5_scores[2], dt_3_scores[2], dt_scores[2]]
R2s   = [dt_full_scores[3], dt_5_scores[3], dt_3_scores[3], dt_scores[3]]
plt.style.use('dark_background')
plt.figure(figsize=(15,8))
plt.subplot(2,2,1)
plt.title("RMSE\n", fontsize=15)
plt.bar(x=DATA_SIZE, height=RMSEs)
plt.axhline(np.mean(RMSEs), color='k', linestyle='--', alpha=0.5, label='Media')
plt.legend()

plt.subplot(2,2,2)
plt.title("MSE\n", fontsize=15)
plt.bar(x=DATA_SIZE, height=MSEs)
plt.axhline(np.mean(MSEs), color='k', linestyle='--', alpha=0.5, label='Media')
plt.legend()

plt.subplot(2,2,3)
plt.title("MAE\n", fontsize=15)
plt.bar(x=DATA_SIZE, height=MAEs)
plt.axhline(np.mean(MAEs), color='k', linestyle='--', alpha=0.5, label='Media')
plt.legend()

plt.subplot(2,2,4)
plt.title("$R^2$\n", fontsize=15)
plt.bar(x=DATA_SIZE, height=R2s)
plt.axhline(np.mean(R2s), color='k', linestyle='--', alpha=0.5, label='Media')
plt.legend()

plt.show()

"""
Al analizar el rendimiento, resulta evidente que ciertas características tienen más peso que otras. Sin embargo, cabe destacar que el conjunto de 
datos completo muestra el rendimiento más superior. Esto se puede atribuir a la información mutua, que revela que cada característica tiene algún 
grado de correlación con la variable objetivo.

En otras palabras, cada característica imparte cierta información a la variable objetivo, contribuyendo así al rendimiento general. Como lo 
demuestra el gráfico de información mutua, todas las características son cruciales para entregar información pertinente a la variable objetivo.

En consecuencia, cuando se utiliza el conjunto de datos completo, el rendimiento es significativamente mejor que cuando se utiliza un conjunto de 
datos reducido. Aunque algunas características parecen más importantes que otras, la presencia de información mutua de todas las características 
en el conjunto de datos completo conduce a un rendimiento superior. Por lo tanto, en este escenario, se debe utilizar el conjunto de datos 
completo para garantizar que se aproveche la información mutua de todas las características para lograr un rendimiento óptimo.

Después de seleccionar nuestro mejor conjunto de funciones, pasemos a seleccionar el mejor modelo.
"""

# Regresión del árbol de decisión
dtr = DTR(max_depth=20, min_samples_leaf=10, min_samples_split=10)

# Modelo de tren
dtr.fit(X_train_full, y_train_full)

# Rendimiento del entrenamiento
y_pred = dtr.predict(X_train_full)
dtr_train_scores = calculate_performance(y_train_full, y_pred)

# Rendimiento de las pruebas
y_pred = dtr.predict(X_test_full)
dtr_test_scores = calculate_performance(y_test_full, y_pred)

# Comparar resultados
plt.figure(figsize=(10,10))
for index, (name, train_score, test_score) in enumerate(zip(SCORE_NAMES, dtr_train_scores, dtr_test_scores)):
    plt.subplot(2,2,index+1)
    plt.title(name, fontsize=12)
    plt.bar(x=name + "(Train)", height=train_score)
    plt.bar(x=name + "(Test)", height=test_score)
plt.show()

# Regresión forestal aleatoria
rfr = RFR(n_estimators=20, min_samples_leaf=10, min_samples_split=10)

# Modelo de tren
rfr.fit(X_train_full, y_train_full)

# Predicciones sobre conjuntos de entrenamiento y prueba
y_pred_train = rfr.predict(X_train_full)
y_pred_test = rfr.predict(X_test_full)

# Calcular el rendimiento del entrenamiento y las pruebas
rfr_train_scores = calculate_performance(y_train_full, y_pred_train)
rfr_test_scores = calculate_performance(y_test_full, y_pred_test)

# Comparar resultados
plt.figure(figsize=(10, 10))
for index, name in enumerate(SCORE_NAMES):
    plt.subplot(2, 2, index+1)
    plt.title(name, fontsize=12)
    plt.bar(x=name + "(Train)", height=rfr_train_scores[index])
    plt.bar(x=name + "(Test)", height=rfr_test_scores[index])
    plt.xticks(rotation=45)
plt.suptitle('Regresión forestal aleatoria\n', fontsize=16)
plt.tight_layout()
plt.show()

"""
Para abordar el problema en cuestión, realicé un análisis exhaustivo de los datos utilizando varios modelos de regresión. Inicialmente, empleé la 
regresión lineal, pero produjo resultados insatisfactorios y, posteriormente, probé la regresión de vectores de soporte, pero resultó ser 
demasiado costosa desde el punto de vista computacional.

Luego, experimenté con la regresión del árbol de decisión y mostró un rendimiento decente. Sin embargo, continué mi búsqueda de un mejor 
rendimiento y finalmente probé la regresión de bosque aleatoria, lo que resultó en una ligera mejora en el rendimiento del modelo.

A pesar del progreso realizado hasta ahora, seguí explorando otras opciones y probé la Regresión de aumento de gradiente, pero desafortunadamente 
no mejoró el rendimiento, sino que lo empeoró.

Después de una evaluación cuidadosa de los resultados, fue evidente que los modelos estadísticos de aprendizaje automático utilizados no eran 
capaces de detectar ningún patrón discernible en los datos.

Después de experimentar con varias arquitecturas de redes neuronales, incluidas las redes neuronales densas, me decepcionó descubrir que no 
funcionaban como se esperaba. A pesar de múltiples intentos, el modelo no pudo alcanzar una puntuación RMSE inferior a 4500.

En un esfuerzo por mejorar el rendimiento, también exploré Conexiones anchas y Omitir, pero desafortunadamente, los resultados no fueron mucho 
mejores. A pesar de su potencial para aumentar la profundidad y amplitud del modelo, estas conexiones no lograron proporcionar el aumento deseado 
en precisión..
"""

# Definir el modelo XGBRegressor
xgb = XGBRegressor(objective='reg:squarederror', colsample_bytree=1, learning_rate=1,
                   max_depth=20, alpha=1, n_estimators=30, subsample=0.5)

# Validación cruzada para evaluación de modelos
xgb_cv_scores = cross_val_score(xgb, X_train_full, y_train_full, cv=5, scoring='neg_mean_squared_error')
xgb_mean_score = np.mean(xgb_cv_scores)
xgb_std_score = np.std(xgb_cv_scores)

# Ajustar el modelo en todo el conjunto de entrenamiento
xgb.fit(X_train_full, y_train_full)

# Predicciones sobre conjuntos de entrenamiento y prueba
y_pred_train = xgb.predict(X_train_full)
y_pred_test = xgb.predict(X_test_full)

# Calcular el rendimiento del entrenamiento y las pruebas
xgb_train_scores = calculate_performance(y_train_full, y_pred_train)
xgb_test_scores = calculate_performance(y_test_full, y_pred_test)

# Comparar resultados
plt.figure(figsize=(10, 10))
for index, name in enumerate(SCORE_NAMES):
    plt.subplot(2, 2, index+1)
    plt.title(name, fontsize=12)
    plt.bar(x=name + "(Train)", height=xgb_train_scores[index])
    plt.bar(x=name + "(Test)", height=xgb_test_scores[index])
    plt.xticks(rotation=45)
plt.suptitle('XGBoost (puntuación CV: {:.2f} +/- {:.2f})\n'.format(xgb_mean_score, xgb_std_score), fontsize=16)
plt.tight_layout()
plt.show()

#El aumento de gradiente extremo (XGBoost) ha demostrado un gran potencial para lograr una alta precisión en problemas de regresión. Sin embargo, 
# para aprovechar plenamente los beneficios de este poderoso algoritmo de aprendizaje automático, es esencial ajustar cuidadosamente sus 
# hiperparámetros para evitar un sobreajuste. El sobreajuste es un desafío común en el aprendizaje automático, donde el modelo es demasiado 
# complejo y se ajusta estrechamente a los datos de entrenamiento, lo que lleva a una mala generalización de datos nuevos e invisibles.

#Para encontrar el conjunto óptimo de hiperparámetros para XGBoost, es necesario equilibrar la complejidad y el rendimiento del modelo. Los 
# hiperparámetros clave, como la tasa de aprendizaje, la profundidad máxima de los árboles y los términos de regularización como alfa y lambda, 
# deben ajustarse cuidadosamente para lograr este equilibrio.

#Ajuste de hiperparámetros
# # Inicializar estimador base
# estimator = XGBRegressor()

# # Inicializar los parámetros de búsqueda (espacio de parámetros)
# params = {
#     'learning_rate':[0.01, 0.1, 1.0],          # En la búsqueda manual de hiperparámetros, 1.0 fue el mejor
#     'colsample_bytree': [0.5,0.8,1],           # Establecer manualmente el valor en 1,0 proporcionaba los mejores resultados, aunque era muy ajustado.
#     'max_depth': [20, 30],                     # Manualmente 20 era razonable
#     'alpha': [1, 5, 10],                       # En mi caso no tuvo un gran impacto.
#     'n_estimators': [30, 50, 100],             # 30 a 50 era razonable.
#     'subsample': [0.5, 0.8, 1.0],              # No tuvo un gran impacto en la reducción del sobreajuste
#     'min_child_weight': [1, 5, 10]
# }

# # Inicializar búsqueda aleatoria
# random_seach = RandomizedSearchCV(
#     estimator=estimator,
#     param_distributions=params,
#     scoring='neg_mean_absolute_error',
#     n_iter=50,                                 # No elegí el 70 porque sé que la mayoría de los parámetros no van a funcionar..
#     cv=5,
#     return_train_score=True,
#     n_jobs=-1,
#     random_state = 123,
#     verbose = 1
# )

# # Empezar a buscar
# random_seach.fit(X_train_full, y_train_full)
# # Consigue el mejor modelo de la búsqueda
# best_model = random_search.best_estimator_

# # Consigue la mejor puntuación obtenida durante la búsqueda
# best_score = random_search.best_score_

# # Obtenga los mejores parámetros encontrados durante la búsqueda
# best_params = random_search.best_params_

# # Imprime los mejores parámetros
# print("Mejores parámetros encontrados durante la búsqueda:", best_params)

# # Imprime la mejor puntuación
# print("Mejor puntuación obtenida durante la búsqueda:", best_score)
#Mejores parámetros encontrados durante la búsqueda: {'subsample': 1.0, 'n_estimators': 100, 'min_child_weight': 10, 'max_depth': 30, 'learning_rate': 0.1, 'colsample_bytree': 1, 'alpha': 5}
#Mejor puntuación obtenida durante la búsqueda: -1811.315517303628

#Mejor modelo:
"""
XGBRegressor(alpha=5, base_score=0.5, booster='gbtree', callbacks=None,
       colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
       early_stopping_rounds=None, enable_categorical=False,
       eval_metric=None, gamma=0, gpu_id=-1, grow_policy='depthwise',
       importance_type=None, interaction_constraints='',
       learning_rate=0.1, max_bin=256, max_cat_to_onehot=4,
       max_delta_step=0, max_depth=30, max_leaves=0, min_child_weight=10,
       missing=nan, monotone_constraints='()', n_estimators=100, n_jobs=0,
       num_parallel_tree=1, predictor='auto', random_state=0, reg_alpha=5)"""

"""El valor de n_iter debe ser lo suficientemente grande como para proporcionar una exploración razonable del espacio de búsqueda de 
hiperparámetros, pero lo suficientemente pequeño como para que sea computacionalmente factible. Una regla general común es establecer n_iter igual 
al número de hiperparámetros que se están ajustando multiplicado por 10. Por ejemplo, si se están ajustando 5 hiperparámetros, un valor de 50 para 
n_iter puede ser un punto de partida razonable.

colsample_bytree: la fracción de columnas que se muestrearán aleatoriamente para cada árbol. Los valores típicos oscilan entre 0,5 y 1.

submuestra: fracción de observaciones que se muestrearán aleatoriamente para cada árbol. Los valores más bajos hacen que el algoritmo sea más 
conservador y reducen el sobreajuste. Los valores típicos oscilan entre 0,5 y 1.

El parámetro min_child_weight es un hiperparámetro en el algoritmo XGBoost que controla la suma mínima de peso de instancia (arpillera) necesaria 
en cada nodo secundario (hoja) de un árbol de decisión. Un valor mayor para min_child_weight alienta al algoritmo a crear árboles con menos nodos 
de hojas, lo que puede ayudar a reducir el sobreajuste al simplificar el modelo.

Las probabilidades mencionadas anteriormente en la distribución de parámetros provienen de la búsqueda manual. Así que puede que no sean los 
mejores, pero sí los mejores que logré manualmente.
"""
# Recrea el mejor modelo (el estado aleatorio es importante).
best_model = XGBRegressor(alpha=5, base_score=0.5, booster='gbtree', callbacks=None,
                          colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
                          early_stopping_rounds=None, enable_categorical=False,
                          eval_metric=None, gamma=0, gpu_id=-1, grow_policy='depthwise',
                          importance_type=None, interaction_constraints='',
                          learning_rate=0.1, max_bin=256, max_cat_to_onehot=4,
                          max_delta_step=0, max_depth=30, max_leaves=0, min_child_weight=10,
                          monotone_constraints='()', n_estimators=100, n_jobs=0,
                          num_parallel_tree=1, predictor='auto', random_state=123, reg_alpha=5, subsample=1.0)

# Entrenar al modelo
best_model.fit(X_train_full, y_train_full)

# Validación cruzada para evaluación de modelos
best_model_cv_scores = cross_val_score(best_model, X_train_full, y_train_full, cv=5, scoring='neg_mean_squared_error')
best_model_mean_score = np.mean(xgb_cv_scores)
best_model_std_score = np.std(xgb_cv_scores)

# Ajustar el modelo en todo el conjunto de entrenamiento
best_model.fit(X_train_full, y_train_full)

# Predicciones sobre conjuntos de entrenamiento y prueba
y_pred_train = best_model.predict(X_train_full)
y_pred_test = best_model.predict(X_test_full)

# Calcular el rendimiento del entrenamiento y las pruebas
best_model_train_scores = calculate_performance(y_train_full, y_pred_train)
best_model_test_scores = calculate_performance(y_test_full, y_pred_test)

# Comparar resultados
plt.figure(figsize=(15, 15))
for index, name in enumerate(SCORE_NAMES):
    plt.subplot(2, 2, index+1)
    plt.title(name, fontsize=12)
    plt.bar(x=name + "(Train)", height=best_model_train_scores[index])
    plt.bar(x=name + "(Test)", height=best_model_test_scores[index])
    plt.xticks(rotation=45)
plt.suptitle('XGBoost (puntuación CV: {:.2f} +/- {:.2f})\n'.format(best_model_mean_score, best_model_std_score), fontsize=16)
plt.tight_layout()
plt.show()

"""
Después de examinar el rendimiento de todos los modelos, es evidente que puede ser necesario un conjunto de datos más grande para identificar 
con precisión el patrón actual y evitar el sobreajuste de los datos. Se observó que casi todos los modelos sobreajustaban los datos hasta cierto 
punto, con la excepción de las redes neuronales profundas. Si bien las redes neuronales profundas no sobreajustaron los datos, su desempeño 
tampoco fue notable. Su error cuadrático medio fue de alrededor de 4000, que es mayor en comparación con los métodos tradicionales de aprendizaje 
automático.

Por otro lado, los métodos tradicionales de aprendizaje automático, como la regresión forestal aleatoria y el árbol de decisión, funcionaron 
considerablemente bien. Ajustando sus parámetros, se puede obtener un modelo robusto. Sin embargo, estos modelos tenían un error cuadrático medio 
de alrededor de 3000 a 4000. Aunque estos modelos mostraron el mismo rendimiento en los datos de entrenamiento y prueba, su rendimiento general no 
fue tan bueno.

Incluso después de hiperajustar los parámetros del modelo de aumento de gradiente extremo, no se pudo obtener un modelo robusto que pueda 
generalizarse bien. Al examinar los datos, descubrimos que la información mutua de todos los vectores de características era crucial. Por lo 
tanto, incluimos todos los datos. Sin embargo, cuando evaluamos la importancia de las características utilizando otros métodos, como el método 
contenedor y el método integrado, obtuvimos resultados que fueron bastante diferentes de los que habíamos obtenido de la información mutua. Esto 
indica que hay algunas relaciones en los datos que no son claramente visibles, lo que hace que los datos en sí sean bastante confusos.

En conclusión, parece que se necesitan más datos para identificar con precisión la selección de características, así como los parámetros del 
modelo. Esto se debe al hecho de que los datos en sí son bastante confusos y las relaciones no son claras.
"""

# mostrar estadísticas resumidas de columnas numéricas
print(df.describe())

# Trazado de cuadros y bigotes de estadísticas resumidas
sns.set_style("darkgrid")
fig, axs = plt.subplots(ncols=3, figsize=(15,15))

sns.boxplot(data=df, x='Duration_in_hours', ax=axs[0], color='skyblue')
axs[0].set_title('Duración en horas\n', fontsize = '16', fontweight = 'bold')

sns.boxplot(data=df, x='Days_left', ax=axs[1], color='salmon')
axs[1].set_title('Días restantes\n', fontsize = '16', fontweight = 'bold')

sns.boxplot(data=df, x='Fare', ax=axs[2], color='lightgreen')
axs[2].set_title('Tarifa\n', fontsize = '16', fontweight = 'bold')

plt.show()

#Aspectos destacados del resumen de estadísticas
#La tarifa promedio de los vuelos es de Rs. 22.840,1, pero la desviación estándar es bastante alta, Rs. 20.307,96 lo que indica un alto grado de 
# variabilidad en los datos. La tarifa mínima para vuelos es de Rs. 1.307, lo que sugiere que también hay algunos vuelos muy baratos disponibles. 
# La tarifa máxima para vuelos es de Rs. 143.019, lo que sugiere que también hay algunos vuelos muy caros. La duración media de los vuelos es de 
# 12,35 horas con una desviación estándar de 7,43 horas, lo que indica que la duración de los vuelos también varía significativamente. La duración 
# mínima de un vuelo es de 0,75 horas, lo que indica que hay algunos vuelos muy cortos en el conjunto de datos. La duración máxima de un vuelo es 
# 43,58 horas, lo que indica que hay algunos vuelos muy largos en el conjunto de datos. El promedio de días restantes para reservar el vuelo es 
# de 25,63 con una desviación estándar de 14,30, lo que significa que la mayoría de las personas reservan sus vuelos con unas 2 o 3 semanas de 
# antelación. En general, podemos ver que existe un alto grado de variabilidad en las tarifas y duraciones de los vuelos, lo que sugiere que puede 
# haber oportunidades para ahorrar dinero siendo estratégico sobre cuándo y dónde reservar vuelos.

#Análisis exploratorio de tarifas
#Tarifa promedio por día de la semana
# Convertir fecha_de_viaje al formato de fecha y hora
df['Date_of_journey'] = pd.to_datetime(df['Date_of_journey'])

# Crear una nueva columna para el día de la semana
df['day_of_week'] = df['Date_of_journey'].dt.day_name()

# Agrupa los datos por día de la semana y calcula la Tarifa media
daily_fares = df.groupby('day_of_week')['Fare'].mean()

#imprimir el resultado
print(daily_fares)

# Crear una nueva columna para el día de la semana
df['day_of_week'] = df['Date_of_journey'].dt.day_name()

# Definir el orden de los días de la semana
day_order = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
cat_dtype = pd.api.types.CategoricalDtype(categories=day_order, ordered=True)

# Convierta la columna day_of_week al tipo de datos categóricos
df['day_of_week'] = df['day_of_week'].astype(cat_dtype)

# Agrupa los datos por día de la semana y calcula la Tarifa media
daily_fares = df.groupby('day_of_week')['Fare'].mean()

# Trazar la serie temporal
plt.plot(daily_fares.index, daily_fares.values, color = "lime", marker = "o")
plt.xlabel('Día de la semana\n')
plt.ylabel('Tarifa\n')
plt.show()

#Podemos ver que los precios son más altos el miércoles, seguido del lunes y sábado. El domingo tiene la cuarta tarifa promedio más alta, seguido 
# de cerca por el martes y el viernes. El jueves tiene la tarifa media más baja en comparación con otros días de la semana.

#Este hallazgo puede resultar útil para los viajeros que buscan ahorrar dinero en sus vuelos. Al reservar sus vuelos el jueves, es posible que 
# puedan ahorrar algo de dinero en comparación con reservar otros días de la semana. Por otro lado, reservar vuelos los miércoles, lunes o sábados 
# puede generar tarifas más altas.

#Es importante tener en cuenta que este análisis se basa en los datos proporcionados y puede no ser necesariamente cierto para todos los vuelos y 
# aerolíneas. Otros factores como la demanda, la estacionalidad y la disponibilidad de asientos también pueden afectar los precios de los vuelos.

#Tarifa promedio versus días restantes
# Agrupa los datos por Días_restantes y calcula la tarifa media
daily_fares = df.groupby('Days_left')['Fare'].mean()

# Trazar la serie temporal
plt.plot(daily_fares.index, daily_fares.values)
plt.xlabel('Días restantes\n')
plt.ylabel('Tarifa\n')
plt.show()

#El Gráfico muestra que hay una ligera disminución en las tarifas a medida que disminuye el número de días que faltan para el viaje. La tarifa es 
# más alta cuando hay solo queda un día para el viaje y disminuye gradualmente a medida que aumentan los días restantes. Sin embargo, esta 
# tendencia no es lineal y existen algunas quedan fluctuaciones en los valores de las tarifas para ciertos días. En general, el número de días 
# restantes parece tener un impacto menor en la tarifa, y otros factores como la demanda, la estacionalidad y la disponibilidad de asientos pueden 
# tener un impacto más significativo en los precios de las tarifas.

#Tarifa promedio del rango de fechas general
# Convertir la columna Fecha_del_viaje a un tipo de fecha y hora
df['Date_of_journey'] = pd.to_datetime(df['Date_of_journey'])

# Agrupa los datos por fecha y calcula la Tarifa media
daily_fare = df.groupby('Date_of_journey')['Fare'].mean()

# Trazar la serie temporal
fig = plt.figure(figsize=(12, 6))
plt.plot(daily_fares.index, daily_fare.values)
plt.xlabel('Fecha\n')
plt.ylabel('Tarifa\n')
plt.show()

#El gráfico anterior muestra que los precios de los vuelos fluctuaron en el transcurso de 50 días con una variación significativa. El 16 de enero 
# la tarifa era # el más alto en 28715,57, y luego disminuyó gradualmente durante los siguientes días. El 25 de enero se produjo otro aumento en 
# las tarifas, seguido de una disminución de los precios. Luego, los precios se mantuvieron relativamente estables hasta el 9 de febrero, cuando 
# cayeron bruscamente hasta su punto más bajo en # 21010.69. No puedo pensar en ningún feriado específico o motivo del aumento de precios en este 
# período. Luego, los precios comenzaron a subir nuevamente hasta febrero.
# 14 y luego estabilizado. El análisis muestra que los precios de los vuelos pueden ser bastante volátiles y cambiar significativamente en un 
# corto período de tiempo.

#Variación de tarifa según la hora de llegada
# Agrupa los datos por Llegada y calcula la Tarifa media
arrival_fares = df.groupby('Arrival')['Fare'].mean().sort_values()

# Trazar el gráfico de barras
plt.bar(arrival_fares.index, arrival_fares.values)
plt.xticks(rotation=45)
plt.xlabel('Llegada\n')
plt.ylabel('Tarifa\n')
plt.show()

#A partir de los gráficos se puede concluir que los precios de los vuelos varían dependiendo de la hora de llegada de los vuelos. Se 
# observan las # tarifas más altas.
# para vuelos que llegan por la tarde (después de las 6:00 p. m.), seguidos de vuelos que llegan por la tarde (de 12:00 p. m. a 6:00 p. m.). 
# Vuelos que llegan por la mañana.
# (6 a.m. - 12 p.m.) tienen una tarifa moderada, mientras que las tarifas más baratas se aplican a los vuelos que llegan antes de las 6 a.m.

#Número de vuelos en cada ruta
# Cuente el número de vuelos para cada par origen-destino
flight_count = df.groupby(['Source', 'Destination']).size().reset_index(name='Count')

# Ordenar los datos por conteo
flight_count = flight_count.sort_values(by='Count')

# Trazar un gráfico de barras del recuento de vuelos por pares origen-destino
fig = plt.figure(figsize=(12, 6))
plt.bar(flight_count['Source'] + '-' + flight_count['Destination'], flight_count['Count'], color = "thistle", edgecolor = "violet")
plt.xticks(rotation=45)
plt.xlabel('Par fuente-destino\n')
plt.ylabel('Número de vuelos\n')
plt.show()

#El mayor número de vuelos se produce entre Delhi-Mumbai con un recuento de 19113, seguido de Bangalore-Delhi con 17636 vuelos. el menor numero
#El número de vuelos se realiza entre Calcuta y Ahmedabad con un recuento de 4791. En general, parece que los vuelos entre las principales 
# ciudades como Delhi, Mumbai,
# Bangalore y Chennai son los más frecuentes.

#Tarifa promedio en cada ruta
# Calcular la tarifa media para cada par origen-destino
mean_fare = df.groupby(['Source', 'Destination'])['Fare'].mean().reset_index(name='Mean Fare')

# Ordenar los datos por tarifa media
mean_fare = mean_fare.sort_values(by='Mean Fare')

# Trazar un gráfico de barras de la tarifa media por pares origen-destino
fig = plt.figure(figsize=(12, 6))
plt.bar(mean_fare['Source'] + '-' + mean_fare['Destination'], mean_fare['Mean Fare'], color = "aqua")
plt.xticks(rotation=45)
plt.xlabel('Par fuente-destino\n')
plt.ylabel('Tarifa promedio\n')
plt.show()

#La ruta con la tarifa promedio más alta es Calcuta-Mumbai con una tarifa promedio de Rs 26997,85. La ruta con la tarifa media más baja es
# Hyderabad-Ahmedabad con una tarifa promedio de 19001,85 rupias. La ruta Ahmedabad-Delhi tiene la tarifa promedio más baja de todas las rutas de 
# Delhi: 19473,74 rupias.
# La tarifa promedio más alta para las rutas de Delhi es Delhi-Hyderabad con Rs19058.78. La ruta con la tarifa media más alta para Bangalore es
# Bangalore-Delhi a 19573,58 rupias. La tarifa más alta para Chennai es Chennai-Hyderabad con 21633,29 rupias. Mumbai-Delhi tiene la tarifa 
# promedio más alta de # todas las rutas de Mumbai a Rs19766,97. La tarifa más alta para Hyderabad es Hyderabad-Delhi con Rs20838,71.

#Mejores y peores días para volar en rutas específicas
# Calcular la tarifa media para cada ruta en cada día de la semana.
df['route'] = df['Source'] + '-' + df['Destination']
route_fares = df.groupby(['route', 'day_of_week'])['Fare'].mean().reset_index()

# Encuentra la ruta con la tarifa máxima para cada día de la semana
max_fares = route_fares.groupby(['day_of_week'])['Fare'].transform(max) == route_fares['Fare']
max_fares = route_fares[max_fares]

# Cree un gráfico de barras para mostrar el par origen-destino con la tarifa promedio más alta en cada día de la semana.
plt.figure(figsize=(15, 15))
for day in df['day_of_week'].unique():
    if day in max_fares['day_of_week'].unique():
        source_dest = max_fares.loc[max_fares['day_of_week'] == day, 'route'].iloc[0]
        plt.bar(day, route_fares.loc[(route_fares['day_of_week'] == day) & (route_fares['route'] == source_dest), 'Fare'].iloc[0], label=source_dest)
plt.legend()
plt.xticks(rotation=45)
plt.xlabel('Día de la semana\n')
plt.ylabel('Tarifa promedio\n')
plt.title('Par Origen-Destino con Tarifa Promedio Máxima en Cada Día de la Semana\n', fontsize = '16', fontweight = 'bold')
plt.show()

# Encuentra la ruta con la tarifa máxima para cada día de la semana
max_fares = route_fares.groupby(['day_of_week'])['Fare'].transform(min) == route_fares['Fare']
max_fares = route_fares[max_fares]

# Cree un gráfico de barras para mostrar el par origen-destino con la tarifa promedio más alta en cada día de la semana
plt.figure(figsize=(15, 15))
for day in df['day_of_week'].unique():
    if day in max_fares['day_of_week'].unique():
        source_dest = max_fares.loc[max_fares['day_of_week'] == day, 'route'].iloc[0]
        plt.bar(day, route_fares.loc[(route_fares['day_of_week'] == day) & (route_fares['route'] == source_dest), 'Fare'].iloc[0], label=source_dest)
plt.legend()
plt.xticks(rotation=45)
plt.xlabel('Día de la semana\n')
plt.ylabel('Tarifa promedio\n')
plt.title('Par fuente-destino con tarifa promedio mínima en cada día de la semana\n', fontsize = '16', fontweight = 'bold')
plt.show()

#Basado en el gráfico de barras anterior, el día más costoso para viajar en la ruta Ahmedabad-Kolkata es el sábado, y en la ruta Ahmedabad-Mumbai.
# El día más costoso es el lunes. Para la ruta Calcuta-Ahmedabad, el martes es el día más caro para viajar. El domingo es el día más caro para viajar 
# en la Ruta Kolkata-Chennai, el viernes es el día más costoso para viajar en la ruta Kolkata-Mumbai, y el jueves es el día más costoso para viajar 
# en la Ruta Mumbai-Calcuta.

#Mientras que, el día más barato para viajar en la ruta Hyderabad-Ahmedabad es el jueves, con una tarifa media de 14647,83. Los días más baratos 
# para los viajes en la ruta Delhi-Hyderabad son los domingos y miércoles, con tarifas promedio de 19152,83 y 19057,66, respectivamente. Por 
# último, el más barato.
# día para viajar en la ruta Bangalore-Ahmedabad es el viernes, con una tarifa media de 16702,53.

#Análisis de correlación
# Función para comprobar la asociación entre variables categóricas con Tarifa
def measure_association(df, fixed_variable, variable):
    # Crear una tabla de contingencia entre las dos variables
    contingency_table = pd.crosstab(df[fixed_variable], df[variable])
    
    # Calcule el estadístico de prueba de chi-cuadrado y el valor P correspondiente
    chi2_stat, p_val, dof, expected = chi2_contingency(contingency_table)
    
    # Compruebe si el valor P es inferior a 0,5
    if p_val < 0.5:
        return "Correlacionado"
    else:
        return "No correlacionado"
#La función anterior medida_association() calcula la asociación entre dos variables categóricas en un conjunto de datos. La función primero crea
# una tabla de contingencia entre las dos variables usando la función pd.crosstab(). Luego, calcula el estadístico de prueba chi-cuadrado, los 
# grados de libertad y las frecuencias esperadas usando la función chi2_contingency(). Finalmente, verifica si el valor P es menor que 0,5, lo 
# que indica una asociación significativa entre las dos variables. Si el valor P es menor que 0,5, devuelve "Correlacionado", de lo contrario 
# devuelve "No correlacionado". La prueba de chi-cuadrado es una prueba estadística utilizada para determinar si existe una asociación 
# significativa entre dos variables categóricas.
# Compara las frecuencias observadas en una tabla de contingencia con las frecuencias esperadas, suponiendo que las dos variables son 
# independientes. Si las frecuencias observadas son significativamente diferentes de las frecuencias esperadas, sugiere que las dos variables no 
# son independientes y están asociados.

# Verificar asociación entre Tarifa y Ruta
association = measure_association(df, "Fare", "route")
print(association)

# Verificar asociación entre tarifa y aerolínea
association = measure_association(df, "Fare", "Airline")
print(association)

# Verificar asociación entre Tarifa y Clase
association = measure_association(df, "Fare", "Class")
print(association)

# Verifique la asociación entre Tarifa y Journey_day
#association = measure_association(df, "Fare", "day_of_week")
#print(association)

# Verificar asociación entre Tarifa y Llegada
association = measure_association(df, "Fare", "Arrival")
print(association)

#Función para comprobar el coeficiente de correlación de la variable numérica
def measure_correlation(data, variable):
    # Calcule el coeficiente de correlación y el valor P entre la tarifa y la variable
    correlation_coef, p_val = stats.pearsonr(data['Fare'], data[variable])
    
    # Compruebe si el valor P es inferior a 0,5
    if p_val < 0.5:
        return f"Correlacionado: coeficiente de correlación = {correlation_coef:.2f}"
    else:
        return "No correlacionado"
# Verificar asociación entre Tarifa y Llegada
correlation = measure_correlation(df,"Duration_in_hours")
print(association)

# Verificar asociación entre Tarifa y Llegada
correlation = measure_correlation(df,"Days_left")
print(association)

# Eliminación de columnas no correlacionadas
# crear una lista de columnas que se eliminarán
columns_to_drop = ['Date_of_journey','Journey_day','Flight_code','Source','Destination']

# soltar las columnas usando la función de soltar
df = df.drop(columns_to_drop, axis=1)
# Realizar una codificación en caliente para preparar la predicción de datos
# Seleccione las columnas categóricas
categorical_cols = ['Airline', 'Class', 'Departure', 'Arrival', 'day_of_week', 'route','Total_stops']

# Realizar codificación one-hot
df_encoded = pd.get_dummies(df, columns=categorical_cols)
df_encoded

#Predicción
#Regresión lineal
# Definir las variables predictoras y objetivo
X = df_encoded.drop('Fare', axis=1)
y = df_encoded['Fare']

# Divida los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear un modelo de regresión lineal y ajustar los datos
model = LinearRegression()
model.fit(X_train, y_train)

# Predecir las tarifas para el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el desempeño del modelo usando R-cuadrado
r2 = r2_score(y_test, y_pred)
print("Puntuación R cuadrado:", r2)

#Bosque aleatorio
# Generar predicciones aleatorias
n_samples = len(X_test)
y_pred = np.random.randint(low=y_train.min(), high=y_train.max()+1, size=n_samples)

# Evaluar el desempeño del modelo usando R-cuadrado
r2 = r2_score(y_test, y_pred)
print("Puntuación aleatoria R cuadrado:", r2)

#Impulso de gradiente
# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicialice el regresor de aumento de gradiente con parámetros predeterminados
gbr = GradientBoostingRegressor()

# Entrene el modelo con los datos de entrenamiento
gbr.fit(X_train, y_train)

# Utilice el modelo entrenado para hacer predicciones sobre los datos de prueba
y_pred = gbr.predict(X_test)

# Evaluar el desempeño del modelo usando R-cuadrado
r2 = r2_score(y_test, y_pred)
print("Puntuación R cuadrado:", r2)

# Divida los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo
#xgb_model = xgb.XGBRegressor(objective='reg:squarederror')
#xgb_model.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
#y_pred = xgb_model.predict(X_test)

# Calcular la puntuación R cuadrado
r2 = r2_score(y_test, y_pred)
print("Puntuación R cuadrado:", r2)

"""Después de realizar varios algoritmos de regresión como regresión lineal, bosque aleatorio, aumento de gradiente y XGBoost en los datos 
proporcionados, descubrió que XGBoost dio la mejor puntuación de R cuadrado de 0,9350. Esto indica que XGBoost es el mejor modelo para predecir 
los precios de las tarifas de vuelos basados ​​en las características dadas.

XGBoost funciona mejor que otros modelos porque es una potente técnica de aprendizaje conjunto que utiliza un algoritmo optimizado de aumento de 
gradiente.
para mejorar la precisión de los clasificadores débiles. Puede manejar bien grandes conjuntos de datos, valores atípicos y valores faltantes. 
XGBoost también proporciona varios hiperparámetros que se pueden ajustar para optimizar aún más el rendimiento del modelo.

Por lo tanto, podemos concluir que XGBoost es el mejor algoritmo de regresión para predecir los precios de las tarifas de los vuelos en función 
de los datos proporcionados y
características."""