import streamlit as st
import pandas as pd
st.title('INDICE DE LA FELICIDAD')
st.subheader('Aquí se observan 10 variables ontemidas del Banco Mundial cuyo proposito es de expicar la felicidad de un país, cuya muestra es de 60 países alrededor del mundo')

data = pd.read_csv('./carpeta/data.csv')
st.header('Los primeros diez países y las respectivas variables')
st.table(data.head(10))

st.header('La descripción de las variables evaluadas en conjunto con la muestra de países')
st.table(data.describe())
