import streamlit as st
import pandas as pd
st.title('INDICE DE LA FELICIDAD')
st.subheader('Aquí se observan 10 variables ontemidas del Banco Mundial cuyo proposito es de expicar la felicidad de un país, cuya muestra es de 60 países alrededor del mundo')

data = pd.read_csv('/carpeta/datasets3.csv')

