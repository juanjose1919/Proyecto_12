import streamlit as st
import pandas as pd
st.title('INDICE DE LA FELICIDAD')
st.subheader('Aquí se observan 10 variables ontemidas del Banco Mundial cuyo proposito es de expicar la felicidad de un país, cuya muestra es de 60 países alrededor del mundo')

data = pd.read_csv('./carpeta/data.csv')
st.header('Los primeros diez países y las variables analizadas:')
st.table(data.head(10))

st.header('La descripción de las variables evaluadas en conjunto con la muestra de países:')
st.table(data.describe())

st.subheader('El algoritmo de reducción de dimensionalidad seleccionado es:)

st.subheader('PCA:')
st.text('Recordemos que es un método estadístico el cual permite simplificar la complejidad de espacios muestrales con muchas dimensiones (10 dimensiones) a la vez que conserva su información')
             

st.table(
'Estandarizar el dataset'
'Construir la matriz de covarianzas'
'Descomponer la matriz de covarianzas en valores propios y vectores propios'
'Seleccionar  vectores propios, los cuales corresponden a los  vectores más grandes, donde  representa la dimensionalidad del nuevo dataset'
'Construir una matriz de proyección , del top de  vectores propios'
'Transformar el dataset input, , utilizando la matriz de proyección  para obtener el nuevo subespacio -dimensional'
)
