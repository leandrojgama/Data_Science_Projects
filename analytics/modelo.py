#Bibliotecas utilizadas

#Tratamento de dados
import pandas as pd
import numpy as np

#Para Machine Learning
from sklearn.model_selection import train_test_split  #para treino dos dados
from sklearn.metrics import classification_report, confusion_matrix #para avaliar o resultados do modelo
from sklearn.inspection import permutation_importance #para avaliar a importancia das variaveis para o modelo
from sklearn.ensemble import RandomForestClassifier #modelo ml de arvores aleatórias

#Fazer a persistencia do modelo na maquina
from joblib import dump,load

#Carregando o arquivo
df= pd.read_csv(r"C:\dev\ml_python\supervisionado\previsao_doencas_cardiaca\datasets\cardio_train.csv", encoding="utf-8", sep=",", index_col=0)

#Transformando a coluna de idade em anos
df['age'] = (df['age']/365).round(0).astype('Int64')


#Separando os dados , valor preditor e variaveis
Y = df["cardio"]
X = df.drop(["cardio", "id"], axis=1)


#Treinando os dados 70% para treino e 30% para treino  
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)

#Random_state apenas para ter uma amostra igual, pode remover!


#instanciando o modelo
ml_model = RandomForestClassifier(n_estimators=20 , n_jobs=4, max_depth=4)

#aplicando o modelo
ml_model.fit(x_train,y_train)

#Aplicando a predição em todos os valores de test
predictions = ml_model.predict(x_test)

#Avaliando os resultados do modelo
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test,predictions))

#Para salvar o modelo
dump(ml_model,'modeloPreditivo1.pK1')
