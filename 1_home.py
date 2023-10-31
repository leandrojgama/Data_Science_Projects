import streamlit as st
import pandas as pd
from datetime import datetime
import webbrowser


st.set_page_config(
    page_title="Home",
    page_icon="🏠",
)

#Carregamento/ tratamento dos dados
if "data" not in st.session_state:
    df= pd.read_csv(r"datasets\cardio_train.csv", encoding="utf-8", sep=",", index_col=0)
    
    #Transformando a coluna de idade em anos
    df['age'] = (df['age']/365).round(0).astype('Int64')
 
    #Para compartilhar o mesmo dataframe entre as páginas
    st.session_state["data"] = df
    
#opção para adicionar textos
st.write("# Previsão de Doenças Cardiacas utilizando Machine Learning ")

#opção para adicionar filtro ou outras informações na parte esquerda da página
st.sidebar.markdown("Dataset - [Kaggle] (https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset?select=cardio_train.csv)")


#opção para adicionar botão com link
bt = st.button("codígo fonte - [github]")
if bt:
    webbrowser.open_new_tab("https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset?select=cardio_train.csv")

#opção para adicionar texto na página formatado
st.markdown(
    """
    Este projeto tem como objetivo realizar a previsão de doenças cardíacas utilizando Machine Learning. 
    Para desenvolver o modelo de previsão, foi considerado como base um conjunto de dados composto por **70 mil registros de pacientes**. 
    Neste projeto, foi utilizado o algoritmo **RandomForestClassifier**, com o qual teve uma **acurácia de 73%**, demonstrando a eficácia da utilização do Machine Learning para prever doenças cardicas.
"""
)

#Video no youtube com exemplo
#.https://www.youtube.com/watch?v=uK_uv59YChk
