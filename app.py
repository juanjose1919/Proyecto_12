import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go

st.title('INDICE DE LA FELICIDAD')
st.subheader('Aquí se observan 10 variables ontemidas del Banco Mundial cuyo proposito es de expicar la felicidad de un país, cuya muestra es de 60 países alrededor del mundo')

data = pd.read_csv('./carpeta/data.csv')
st.header('Los primeros diez países y las variables analizadas:')
st.table(data.head(10))

st.header('La descripción de las variables evaluadas en conjunto con la muestra de países:')
st.table(data.describe())

st.subheader('El algoritmo de reducción de dimensionalidad seleccionado es:')

st.subheader('PCA:')
st.write('Recordemos que es un método estadístico el cual permite simplificar la complejidad de espacios muestrales con muchas dimensiones (10 dimensiones) a la vez que conserva su información')
             
st.subheader('Los pasos a seguir son:')

st.write('1. Estandarizar el dataset')
code1 = '''lista = []
for i in X.T:
  u = i.mean()
  s = i.std()
  scal = (i - u) / s
  lista.append(scal)

x_scal = np.array(lista).T'''
st.code(code1, language='python')

st.write('2. Construir la matriz de covarianzas')
code2 = '''cov_x = np.cov(x_scal.T)
cov_x'''
st.code(code2, language='python')

st.write('3. Descomponer la matriz de covarianzas en valores propios y vectores propios')
code3 = '''cov_x = np.cov(x_scal.T)
cov_x
np.linalg.eig(pd.DataFrame(X).corr().to_numpy())'''
st.code(code3, language='python')

st.write('4. Seleccionar  vectores propios, los cuales corresponden a los  vectores más grandes, donde  representa la dimensionalidad del nuevo dataset')
code4 = ''' 
val_p, vec_p = linalg.eig(cov_x)
val_p, vec_p

# 3 componentes

val_p = val_p[:3]
vec_p = vec_p[:, :3]
'''
st.code(code4, language='python')

st.write('5. Construir una matriz de proyección , del top de  vectores propios')
code5 = ''' 
W = vec_p

# proyectar X en W
pca_p = x_scal @ W
'''
st.code(code5, language='python')

st.write('6. Transformar el dataset input, , utilizando la matriz de proyección  para obtener el nuevo subespacio -dimensional')
code6 = ''' 
pca_p = pd.DataFrame(pca_p, columns=[f'PC{i}' for i in range(1, pca_p.shape[1] + 1)])
pca_p.head()
'''
st.code(code6, language='python')

pca_p = pd.read_csv('./carpeta/pca_p.csv')
st.header('Los primeros 10 países reducidos a 3 PCA:')
st.table(pca_p.head(10))
         
st.header('La descripción de los componentes principales en conjunto con la muestra de países:')
st.table(pca_p.head(10))

st.header('Grafica: PCA')


for m, zlow, zhigh in [('o', -50, -25), ('^', -30, -5)]:
    xs = pca_p.iloc[:,0]
    ys = pca_p.iloc[:,1]
    zs = pca_p.iloc[:,2]
scene = dict(xaxis = dict(title = 'PCA1'), yaxis = dict(title = 'PCA2'), zaxis= dict(title = 'PCA3'))
trace1 = [go.Scatter3d(x = xs, y= ys, z= zs)]

fig = go.Figure(trace1, layout=go.Layout(margin = dict(l = 0, r = 0),scene = scene, height = 800, width = 800))
fig.show()

  
  
