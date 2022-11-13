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

st.subheader('Número de componentes optimos')
st.image('./carpeta/Num_componentes.png')

st.header('Grafica: PCA')


for m, zlow, zhigh in [('o', -50, -25), ('^', -30, -5)]:
    xs = pca_p.iloc[:,0]
    ys = pca_p.iloc[:,1]
    zs = pca_p.iloc[:,2]
scene = dict(xaxis = dict(title = 'PCA1'), yaxis = dict(title = 'PCA2'), zaxis= dict(title = 'PCA3'))
trace = [go.Scatter3d(x = xs, y= ys, z= zs)]

fig = go.Figure(trace, layout=go.Layout(margin = dict(l = 0, r = 0),scene = scene, height = 800, width = 800))
fig.show()
st.plotly_chart(fig,use_container_widht=True)

code7 = ''' 
sklearn_loadings = pca3.components_.T * np.sqrt(pca3.explained_variance_)
sklearn_loadings'''
st.code(code7, language='python')  
  
st.image('./carpeta/Aportes_a_PC1.png') 
st.image('./carpeta/Aportes_a_PC2.png')
st.image('./carpeta/Aportes_a_PC3.png')


st.header('K-means')
code8 = '''plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')'''
st.code(code8, language='python')

code9 = '''
#kmenas with PCA
X = np.array(pca_p)
y = np.array(pca_p)
#hllamos el punto de codo del valor k
Nc = range(1, 10)
kmeans = [KMeans(n_clusters=i) for i in Nc]
kmeans
score = [kmeans[i].fit(X).score(X) for i in range(len(kmeans))]
score
plt.plot(Nc,score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()
'''
st.code(code9, language='python')

st.image('./carpeta/Numero_optimo.png')

code10 = '''
kmeans = KMeans(n_clusters=4).fit(X)
centroids = kmeans.cluster_centers_
centroids
'''
st.code(code10,  language='python')

code11 = '''
#kmeans with PCA
# Predicting the clusters
labels = kmeans.predict(X)
# Getting the cluster centers
C = kmeans.cluster_centers_
colores=['blue','cyan','green','red']
asignar=[]
for row in labels:
  asignar.append(colores[row])

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=asignar,s=60)
ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c=colores, s=1000)
'''
st.code(code11,  language='python')

st.image('./carpeta/kmeans_PCA.png')
st.write('En el gráfico anterior se puede observar cuatro centroides, entre estos el grupo celeste es el más diferenciado pues presenta mayor distancia con el resto de los datos, para el grupo rojo la agrupación se vuelve menos clara y se evidencian datos erróneamente clasificados que podrían pertenecer a otros grupos. Por último los dos grupos restantes, verde y azul oscuro, no presentan distanciamiento significativo entre sus datos y la participación de grupos se hace menos clara')



code12= '''
#calcular distancia euclidiana
def dis_eucl(a, b):
  return np.sqrt(sum((a - b)**2))
  
#escoger los centroides, k=num_centroides
centroides = {} #centroides
k = 4 #num_centroides
num_iteraciones = 200
for i in range(k):
  centroides[i] = X[np.random.choice(len(X))]

for i in range(num_iteraciones):
  #distancias
  distances = {}
  for pos, dato in enumerate(X):
    distances[pos] = []
    for _ , centroide in centroides.items():
      distances[pos].append(dis_eucl(centroide, dato))

  #asignación
  puntos_centroides = {}

  for i in range(k):
    puntos_centroides[i] = []

  for dato, dists in distances.items():
    puntos_centroides[dists.index(min(dists))].append(X[dato])

  #nuevos centroides

  for cent, datos in puntos_centroides.items():
    centroides[cent] = np.average(datos, axis=0)
    
    
x_0 = np.vstack(puntos_centroides[0])
x_1 = np.vstack(puntos_centroides[1])
x_2 = np.vstack(puntos_centroides[2])
x_3 = np.vstack(puntos_centroides[3])
'''
st.code(code12 , language='python')

code13 ='''
#kmenas with PCA
# Predicting the clusters
labels = kmeans.predict(X)
# Getting the cluster centers
C = kmeans.cluster_centers_
colores=['blue','cyan','green','red']
asignar=[]
for row in labels:
  asignar.append(colores[row])

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x_0[:,0],x_0[:,1])
ax.scatter(x_1[:,0],x_1[:,1])
ax.scatter(x_2[:,0],x_2[:,1])
ax.scatter(x_3[:,0],x_3[:,1])
ax.scatter(centroides[0][0],centroides[0][1], marker='x',color='k')
ax.scatter(centroides[1][0],centroides[1][1], marker='x',color='k')
ax.scatter(centroides[2][0],centroides[2][1], marker='x',color='k')
ax.scatter(centroides[3][0],centroides[3][1], marker='x',color='k')
'''
st.code(code13, language='python')

st.image('./carpeta/otra.png')
st.write('En este gráfico se presentan cuatro centroides y cuatro grupos diferenciados, los datos del grupo azul se encuentran a mayor distancia de los demás grupos, los tres restantes presentan menor distancia entre ellos pero aún así se evidencia una partición clara y una distancia correspondiente entre sus centroides')

code14 ='''
# Obtener los valores y trazarlos PCA1 Y PCA2
f1 = pca_p['PCA1'].values
f2 = pca_p['PCA2'].values

plt.scatter(f1, f2, c=asignar, s=70)
plt.scatter(C[:, 0], C[:, 1], marker='*', c=colores, s=1000)
plt.show()
'''
st.code(code14, language='python')
st.image('./carpeta/una.png')

code15 ='''
# Obtener los valores y trazarlos PCA1 Y PCA3
f1 = pca_p['PCA1'].values
f2 = pca_p['PCA3'].values

plt.scatter(f1, f2, c=asignar, s=70)
plt.scatter(C[:, 0], C[:, 1], marker='*', c=colores, s=1000)
plt.show()
'''
st.code(code15, language='python')
st.image('./carpeta/dos.png')
st.write('En el anterior gráfico se generan cuatro centroides sin una partición completamente clara entre los grupos, en el caso del grupo celeste tiene datos dispersos y el centroide está distanciado, el grupo azul oscuro tiene menor distanciamiento pero en el caso de el rojo y el verde no se ve una clara agrupación, los datos de ambos están ubicados alrededor del centroide rojo mientras el centroide verde se encuentra a una distancia bastante significativa.')

code16 ='''
# Obtener los valores y trazarlos PCA2 Y PCA3
f1 = pca_p['PC2'].values
f2 = pca_p['PC3'].values

plt.scatter(f1, f2, c=asignar, s=70)
plt.scatter(C[:, 0], C[:, 1], marker='*', c=colores, s=1000)
plt.show()
'''
st.code(code16, language='python')
st.image('./carpeta/tres.png')
st.write('En este gráfico tampoco se presenta un agrupamiento diferenciado pues ninguno de los grupos se encuentra ubicado correctamente entorno a su centroide. El centroide verde se encuentra a una distancia significativa mientras que la mayoría de sus datos se ubican al rededor del centrado azul, en torno al centroide rojo se encuentra el grupo rojo junto a varios datos de los demas grupos clasificados erróneamente, por último algunos datos del grupo celeste se encuentran mas cercanos a su centroide mientras la mayoría están dispersos entre los otros grupos.')

st.header('davies_bouldin_score')

code17 ='''
results = {}

for i in range(2,11):
    kmeans = KMeans(n_clusters=i, random_state=30)
    labels = kmeans.fit_predict(X)
    db_index = davies_bouldin_score(X, labels)
    results.update({i: db_index})
'''
st.code(code17, language='python')

code18 ='''
plt.plot(list(results.keys()), list(results.values()))
plt.xlabel("Number of clusters")
plt.ylabel("Davies-Boulding Index")
plt.show()
'''
st.code(code18, language='python')

st.image('./carpeta/davies_bouldin_score.png')
