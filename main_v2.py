# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 22:54:08 2020

@author: acamargo
"""

#import numpy as np
import pandas as pd


#from sklearn.feature_selection import SelectKBest

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

from sklearn import metrics


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

file = 'C:\MLGIS\Datav2ML.csv'


##############################################################################
# funciones
##############################################################################
# funcion para scla_uso
def f_scla_uso(v_scla_uso):
    if  v_scla_uso == 'AGRICOLA':
        return 0
    if  v_scla_uso == 'AGROPECUARIO':
        return 1
    if  v_scla_uso == 'COMERCIO':
        return 2
    if  v_scla_uso == 'EDUCACION':
        return 3
    if  v_scla_uso == 'ERIAZO':
        return 4
    if  v_scla_uso == 'FORESTAL':
        return 5
    if  v_scla_uso == 'INDUSTRIA':
        return 6
    if  v_scla_uso == 'OTROS USOS':
        return 7
    if  v_scla_uso == 'RECREACION PUBLICA':
        return 8
    if  v_scla_uso == 'SALUD':
        return 9
    if  v_scla_uso == 'VIVIENDA':
        return 10
    if  v_scla_uso == 'VIVIENDA - COMERCIO':
        return 11
    else :
        return 999


# funcion para clas_uso
def f_clas_uso(v_clas_uso):
    if  v_clas_uso == 'SUELO PREDOMINANTEMENTE RESIDENCIAL':
        return 0
    if  v_clas_uso == 'SUELO DEDICADO A EQUIPAMIENTOS':
        return 1
    if  v_clas_uso == 'SUELO INDUSTRIAL':
        return 2
    if  v_clas_uso == 'SUELO PREDOMINANTEMENTE COMERCIAL':
        return 3
    if  v_clas_uso == 'SUELO AGRICOLA':
        return 4
    if  v_clas_uso == 'SUELO ERIAZO':
        return 5
    if  v_clas_uso == 'SUELO AGROPECUARIO':
        return 6
    if  v_clas_uso == 'SUELO FORESTAL':
        return 7
    else :
        return 999

# funcion para descrip -> descripcion de la inclinacion del suelo
def f_descrip(v_descrip):
    if  v_descrip == 'PLANO O LIGERAMENTE INCLINADO':
        return 0
    if  v_descrip == 'MODERADAMENTE INCLINADO':
        return 1
    if  v_descrip == 'MODERADAMENTE EPINADO':
        return 2
    else :
        return 999
    
def f_niv_riesgo_pluvial(v_niv_riesgo_pluvial):
    if  v_niv_riesgo_pluvial == 'BAJO':
        return 0
    if  v_niv_riesgo_pluvial == 'MEDIO':
        return 1
    if  v_niv_riesgo_pluvial == 'ALTO':
        return 2
    if  v_niv_riesgo_pluvial == 'MUY ALTO':
        return 3
    else :
        return 999

def f_cobertura_servicios(v_cobertura_servicios):
    if  v_cobertura_servicios == 'SIN COBERTURA':
        return 0
    if  v_cobertura_servicios == 'CON COBERTURA':
        return 1
    else :
        return 999
    
def f_vial_sup_viaL(v_vial_sup_vial):
    if  v_vial_sup_vial == 'AFIRMADO':
        return 0
    if  v_vial_sup_vial == 'ASFALTADA':
        return 1
    else :
        return 999
   
def standarize_z_value(vector):
    return (vector - vector.mean())/vector.std()

# Leyendo la data
df_data_in = pd.read_csv(file)

# seleccionando el campo scla_uso como variable dependiente
# df_y = df_data_in[['scla_uso']]

# print('Unique values of df_y scla_uso', list(df_y.scla_uso.unique()))
# print('Unique values of df_y scla_uso', df_y.scla_uso.value_counts())
# df_pre_y = df_y.applymap(f_scla_uso)


# print('Unique values of df_y scla_uso', list(df_pre_y.scla_uso.unique()))
# print('Unique values of df_y scla_uso', df_pre_y.scla_uso.value_counts())
#df_y.apply(f_scla_uso)


# seleccionado los campos que vendran a ser las caracteristicas iniciales
df = df_data_in[['clas_uso', 'scla_uso', 'area','perimetro' , 'st_x_centroide', 
                 'st_y_centroide','pob_tot07', 'pob_tot17' ,'pob_tot20', 'descrip',
                 'niv_riesgo_sismo', 'niv_riesgo_pluvial', 
                 'niv_riesgo_fluvial', 'cob_rs17', 'cob_ap17', 'cob_de17', 
                 'cob_ee17', 'rango_vs', 'vial_sup_via', 'distancia1']]

# Ver la stadistica de la data usando describe()
df_stat = df.describe()




# counting the values of clas_uso
print('Unique values of df_y clas_uso \n', list(df.clas_uso.unique()))
print('Unique values of df_y clas_uso \n',  df.clas_uso.value_counts())

# transformando los valores de class_uso a numeros enteros de 0 a 7 usnando la
# funcion f_clas_uso

print('Unique values of  scla_uso', list(df.scla_uso.unique()))
print('Unique values of  scla_uso', df.scla_uso.value_counts())
df.clas_uso = df.clas_uso.apply(f_clas_uso)

# area
print('NULLs in area \n', df.area.isnull().sum())
print('mean value of area : ', df.area.mean())
print('std value of area : ', df.area.std())


#scla_uso, luego lo puedo sacar como variable dependiente

df.scla_uso = df.scla_uso.apply(f_scla_uso)


# z - transform ( normalizando )
df.area = ( df.area - df.area.mean() ) / ( df.area.std())

# perimetro
print('NULLs in perimetro \n', df.perimetro.isnull().sum())
print('mean value of perimetro : ', df.perimetro.mean())
print('std value of perimetro : ', df.perimetro.std())

# z - transform ( normalizando )
df.perimetro = ( df.perimetro - df.perimetro.mean() ) / ( df.perimetro.std())

# st_x_centroide
print('NULLs in st_x_centroide \n', df.st_x_centroide.isnull().sum())
print('mean value of st_x_centroide : ', df.st_x_centroide.mean())
print('std value of st_x_centroide : ', df.st_x_centroide.std())

# z - transform ( normalizando )
df.st_x_centroide = ( df.st_x_centroide - df.st_x_centroide.mean() ) / ( df.st_x_centroide.std())

# st_y_centroide
print('NULLs in st_y_centroide \n', df.st_y_centroide.isnull().sum())
print('mean value of st_y_centroide : ', df.st_y_centroide.mean())
print('std value of st_y_centroide : ', df.st_y_centroide.std())

# z - transform ( normalizando )
df.st_y_centroide = ( df.st_y_centroide - df.st_y_centroide.mean() ) / ( df.st_y_centroide.std())

# pob_tot07
print('NULLs in pob_tot07 \n', df.pob_tot07.isnull().sum())
print('mean value of pob_tot07 : ', df.pob_tot07.mean())# np.nanmean(df.pob_tot07))
print('std value of pob_tot07 : ', df.pob_tot07.std())

# z - transform ( normalizando )
df.pob_tot07 = ( df.pob_tot07 - df.pob_tot07.mean() ) / ( df.pob_tot07.std())

# pob_tot17
print('NULLs in pob_tot17 \n', df.pob_tot17.isnull().sum())
print('mean value of pob_tot17 : ', df.pob_tot17.mean())# np.nanmean(df.pob_tot07))
print('std value of pob_tot17 : ', df.pob_tot17.std())

# z - transform ( normalizando )
df.pob_tot17 = ( df.pob_tot17 - df.pob_tot17.mean() ) / ( df.pob_tot17.std())

# pob_tot20
print('NULLs in pob_tot20 \n', df.pob_tot20.isnull().sum())
print('mean value of pob_tot20 : ', df.pob_tot20.mean())# np.nanmean(df.pob_tot07))
print('std value of pob_tot20 : ', df.pob_tot20.std())

# z - transform ( normalizando )
df.pob_tot20 = ( df.pob_tot20 - df.pob_tot20.mean() ) / ( df.pob_tot20.std())


# descrip : es la descripcion de la inclinacion del suelo
print('Unique values of df.descrip  \n', list(df.descrip.unique()))
print('Value counts of df.descrip \n',  df.descrip.value_counts())

# transformando los valores de descrip a numeros enteros de 0 a 2 usando la
# funcion f_descrip

df.descrip = df.descrip.apply(f_descrip)

# niv_riesgo_sismo
print('Unique values of df.niv_riesgo_sismo  \n', list(df.niv_riesgo_sismo.unique()))
print('Value counts of df.niv_riesgo_sismo \n',  df.niv_riesgo_sismo.value_counts())

# transformando los valores de descrip a numeros enteros de 0 a 2 usando la
# funcion f_niv_riesgo_pluvial ya que tienen los mismos niveles
# BAJO, MEDIO, ALTO, MUY ALTO

# f_niv_riesgo_pluvial
df.niv_riesgo_sismo = df.niv_riesgo_sismo.apply(f_niv_riesgo_pluvial) 


# niv_riesgo_masas
# print('Unique values of df.niv_riesgo_masas  \n', list(df.niv_riesgo_masas.unique()))
# print('Value counts of df.niv_riesgo_masas \n',  df.niv_riesgo_masas.value_counts())


# nivel riesgo pluvial niv_riesgo_pluvial
print('Unique values of df.niv_riesgo_pluvial  \n', list(df.niv_riesgo_pluvial.unique()))
print('Value counts of df.niv_riesgo_pluvial \n',  df.niv_riesgo_pluvial.value_counts())

# funcion f_niv_riesgo_pluvial ya que tienen los mismos niveles
# BAJO, MEDIO, ALTO, MUY ALTO

df.niv_riesgo_pluvial = df.niv_riesgo_pluvial.apply(f_niv_riesgo_pluvial)

# nivel riesgo pluvial niv_riesgo_fluvial
print('Unique values of df.niv_riesgo_fluvial  \n', list(df.niv_riesgo_fluvial.unique()))
print('Value counts of df.niv_riesgo_fluvial \n',  df.niv_riesgo_fluvial.value_counts())

# funcion f_niv_riesgo_pluvial ya que tienen los mismos niveles
# BAJO, MEDIO, ALTO, MUY ALTO

df.niv_riesgo_fluvial = df.niv_riesgo_fluvial.apply(f_niv_riesgo_pluvial)

# cob_rs17

print('Unique values of df.cob_rs17  \n', list(df.cob_rs17.unique()))
print('Value counts of df.cob_rs17 \n',  df.cob_rs17.value_counts())

# funcion f_cobertura_servicios para cob_rs17
# 

df.cob_rs17 = df.cob_rs17.apply(f_cobertura_servicios)

# cob_ap17

print('Unique values of df.cob_ap17  \n', list(df.cob_ap17.unique()))
print('Value counts of df.cob_ap17 \n',  df.cob_ap17.value_counts())

# funcion f_cobertura_servicios para cob_ap17
# 

df.cob_ap17 = df.cob_ap17.apply(f_cobertura_servicios)


#  cob_de17
# 

print('Unique values of df.cob_de17  \n', list(df.cob_de17.unique()))
print('Value counts of df.cob_de17 \n',  df.cob_de17.value_counts())

# funcion f_cobertura_servicios para cob_de17
# 

df.cob_de17 = df.cob_de17.apply(f_cobertura_servicios)

# cob_ee17

print('Unique values of df.cob_ee17  \n', list(df.cob_ee17.unique()))
print('Value counts of df.cob_ee17 \n',  df.cob_ee17.value_counts())

# funcion f_cobertura_servicios para cob_ee17
# 

df.cob_ee17 = df.cob_ee17.apply(f_cobertura_servicios)

# rango_vs

print('Unique values of df.rango_vs  \n', list(df.rango_vs.unique()))
print('Value counts of df.rango_vs \n',  df.rango_vs.value_counts())

# Esta variable tiene muchos NULLs, por lo tanto lo vamos a eliminar

df = df.drop(['rango_vs'], axis=1)

# vial_sup_via
print('Unique values of df.vial_sup_via  \n', list(df.vial_sup_via.unique()))
print('Value counts of df.vial_sup_via \n',  df.vial_sup_via.value_counts())


df.vial_sup_via = df.vial_sup_via.apply(f_vial_sup_viaL)


# distancia1
print('NULLs in distancia1 \n', df.distancia1.isnull().sum())
print('mean value of distancia1 : ', df.distancia1.mean())# np.nanmean(df.pob_tot07))
print('std value of distancia1 : ', df.distancia1.std())

# z - transform ( normalizando )
df.distancia1 = ( df.distancia1 - df.distancia1.mean() ) / ( df.distancia1.std())

# info de df

df.info()

# Doing some plots
import seaborn as sns
sns.boxplot(x=df['area'])

import matplotlib.pyplot as plt
import seaborn as sns
correlations = df.corr()
sns.heatmap(data = correlations, square = True, cmap = "bwr")

plt.yticks(rotation = 0)
plt.xticks(rotation = 90)


# box plot for each column
#%matplotlib inline
import matplotlib.pyplot as plt

for column in df:
    plt.figure()
    df.boxplot([column])
    
    



df.clas_uso.isnull().sum()
df.pob_tot07.isnull().sum()

# Detectando missing values (NULLS)
df.isna().any()

# Trabajando con la imputacion de la data NULL

# from sklearn.preprocessing import Imputer
# mean_imputer = Imputer(missing_values = np.nan, strategy='mean', axis = 1)
# mean_imputer = mean_imputer.fit(df)

# imputed_df = mean_imputer.transform(df.values)
# df = pd.DataFrame(data=imputed_df, columns = cols)

df1 = df.dropna()
df1.isna().any()

# Ahora transformando la variable dependiente en variable binaria
df_Y = df1.scla_uso
df_Y[df_Y != 10] = 0


df_Y[df_Y == 10] = 1

df_Y.value_counts()

##############################################################################
# Ahora comenzamos con los modelos de machine learning
##############################################################################

# creando la data de prueba y de entrenamiento

#scla_uso es usado como variable dependiente, por lo que debemos removerla del df1

df_X = df1.drop(['scla_uso'], axis = 1)

X_values = df_X.values
Y_values = df_Y.values


X_train, X_test, y_train, y_test = train_test_split(X_values, Y_values, test_size = 0.70 ,random_state = 45)

##############################################################################
# logistic regression
##############################################################################

logisticRegres = LogisticRegression(max_iter = 1000)
logisticRegres.fit(X_train, y_train)

y_predict = logisticRegres.predict(X_test)



score = logisticRegres.score(X_test, y_test)
print("Precision Logistic regression :", score)
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import mean_absolute_error

print(logisticRegres.coef_[0])
print(logisticRegres.intercept_[0])

coef_lr = logisticRegres.coef_
# Printing the Mean absolute error and mean square error
mean_absolute_error(y_test, y_predict)
mean_squared_error(y_test, y_predict)

# from sklearn.metrics import roc_auc_score
cm_lr = confusion_matrix(y_test, y_predict.astype('int'), labels=  logisticRegres.classes_)
roc_auc_score(y_test, y_predict)

##############################################################################
# random forest
##############################################################################

clf=RandomForestClassifier(n_estimators=256)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred_rf =clf.predict(X_test)
y_rf_roc = clf.predict_proba(X_test)[:,1]
#Import scikit-learn metrics module for accuracy calculation

# Model Accuracy
print("Precision :",metrics.accuracy_score(y_test, y_pred_rf))


# Printing the Mean absolute error and mean square error
mean_absolute_error(y_test, y_pred_rf)
mean_squared_error(y_test, y_pred_rf)

auc_rf = roc_auc_score(y_test, y_pred_rf)
cm_rf = confusion_matrix(y_test, y_pred_rf, labels=  clf.classes_)
roc_auc_score(y_test, y_pred_rf)
