##Bibliotecas utilizadas##

#Criação da aplicação web
import streamlit as st

#Tratamento de dados
import pandas as pd
import numpy as np

#Visualização de dados (gráficos)
import altair as alt


#Machine Learning
from sklearn.model_selection import train_test_split  #para treino dos dados
from sklearn.metrics import classification_report, confusion_matrix #para avaliar o resultados do modelo
from sklearn.inspection import permutation_importance #para avaliar a importancia das variaveis para o modelo
from sklearn.ensemble import RandomForestClassifier #modelo ml de arvores aleatórias
from sklearn.inspection import permutation_importance #importancia das variaveis
from sklearn.metrics import roc_curve, roc_auc_score #curva roc


##Carregando os dados de home##
df = st.session_state["data"]

##Gerando o modelo de machine Learning##
ml_model = RandomForestClassifier(n_estimators=20, n_jobs=4, max_depth=4)
       
# Separando os dados em valor preditor e variáveis
Y = df["cardio"]
X = df.drop(["cardio", "id"], axis=1)

# Treinando os dados com 70% para treino e 30% para teste
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)

# Instanciando o modelo
ml_model.fit(x_train, y_train)

#Aplicando a predição em todos os valores de test
predictions = ml_model.predict(x_test)

##Gerando os resultados do modelo (Precisão, recall, F1)##
classification_rep = classification_report(y_test, predictions, output_dict=True)
df_classificacao = pd.DataFrame(classification_rep).transpose()

#Removendo a coluna support (não será utilizada no gráfico)
df_classificacao.drop('support', axis=1, inplace=True)

#Convertendo as métricas para porcentagem (precision, recall e f1-score)
df_classificacao['precision'] = (df_classificacao['precision'] * 100).apply(lambda x: f'{x:.2f}%')
df_classificacao['recall'] = (df_classificacao['recall'] * 100).apply(lambda x: f'{x:.2f}%')
df_classificacao['f1-score'] = (df_classificacao['f1-score'] * 100).apply(lambda x: f'{x:.2f}%')


#Gerando a matrix de confusão
df_matrix= (confusion_matrix(y_test,predictions))

#df_matrix = pd.DataFrame(df_matrix, columns=['Verdadeiro Positivo', 'Falso Negativo'])
df_matrix = pd.DataFrame(df_matrix, columns=["True positive", "False negative"], index=["False positive", "True negative"])


#Calculando a importancia das variaveis
result = permutation_importance(ml_model, x_test, y_test, n_repeats=10, n_jobs=2)
sorted_idx = result.importances_mean.argsort()

importances = result.importances_mean
features = x_test.columns 

df_importance = pd.DataFrame({'Variaveis': features, 'Importancia': importances})


# Calculando a matrix de correlação
corr_matrix = df.corr()

# Transformando a matriz de correlação em um formato adequado para utilizar no gráfico
corr_matrix = corr_matrix.unstack().reset_index()
corr_matrix.columns = ['Variveis 1', 'Variveis 2', 'Correlation']


# Calculando as probabilidades previstas pelo modelo
y_probs = ml_model.predict_proba(x_test)[:, 1]

# Calculando a curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# Criando um DataFrame com os dados da curva ROC
roc_df = pd.DataFrame({'False Positive Rate': fpr, 'True Positive Rate': tpr})


##Desenvolvendo a aplicação WEB##

#Header da página
st.header('Resultados do modelo')

col1, col2 = st.columns(2)

#Tabela com os resultados do modelo
with col1:
    st.write('Resultado do modelo')
    st.dataframe(df_classificacao)

#Tabela com a matrix de confusão
with col2:
    st.write('Matriz de confusão')
    st.dataframe(df_matrix)

col1, col2 = st.columns(2)

#Gráfico de importância das variáveis  
with col1:
    st.write('Importância das variáveis ')   
    chart = alt.Chart(df_importance).mark_bar().encode(
        x='Importancia:Q',
        y=alt.Y('Variaveis:N', sort='-x')
    ).properties(
        width=400,
        height=300
    )

    chart
    
#Gráfico de correlação entre as variaveis
with col2:
    st.write('Correlaçao entre variáveis')  
    heatmap = alt.Chart(corr_matrix).mark_rect().encode(
        x='Variveis 1:N',
        y='Variveis 2:N',
        color='Correlation:Q'
    ).properties(
        width=550,
        height=400
    )

    heatmap

#Grafico curva ROC
st.write('Curva ROC')

st.altair_chart(
    alt.Chart(roc_df)
    .mark_line()
    .encode(
        x='False Positive Rate:Q',
        y='True Positive Rate:Q'
    )
    .configure_mark(color='blue')
    .configure_axis(labelFontSize=12, titleFontSize=14)
    .properties(width=800, height=400)
)

#Tabela utilizada para criar o modelo
st.markdown('Tabela com os resultados utilizados')

st.dataframe(df,use_container_width=True)
