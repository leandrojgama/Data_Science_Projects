#Bibliotecas

#Criação da aplicação web
import streamlit as st

#Tratamento de dados
import numpy as np
import pandas as pd

#Machine Learning
import sklearn
from sklearn.model_selection import train_test_split  #para treino dos dados
from sklearn.metrics import classification_report, confusion_matrix #para avaliar o resultados do modelo
from sklearn.inspection import permutation_importance #para avaliar a importancia das variaveis para o modelo
from sklearn.ensemble import RandomForestClassifier #modelo ml de arvores aleatórias


#Desenvolvimento da aplicação web
#Configuração da página
st.set_page_config(
    page_title="Realizar previsão"
)


#Header
st.markdown('Aplicação do modelo de previsão de doença cardiaca')


#Inserir o nome (input de informações)
nome = st.text_input('Informe seu nome:')
if nome:  # Verifica se o campo 'nome' não está vazio
    st.write('Olá,', nome, '!')
    st.write('Fique tranquilo! Seus dados são confidenciais e não serão armazenados nem compartilhados. ')
    
st.header('Insira as informações para efetuar a previsão')

#Inserir idade (input de informações)
idade = st.number_input('Informe sua idade:', step=1)

#Filtro de opções para sexo (input de informações)
lista_sexo = ['-','Feminino', 'Masculino']
sexo = st.radio('Informe seu sexo:', options=lista_sexo)
result_sex = {'-':None,'Feminino': 1, 'Masculino': 2}
sexo_resultado = result_sex.get(sexo, None)

#peso (em Kilos - número) (input de informações)
peso = st.number_input('Informe seu peso em kilos (kg):')

#altura (em centimetros - número) (input de informações)
altura = st.number_input('Informe sua altura em centímetros (cm):')

#pressoa arterial sistoica (número) (input de informações)
pressao_sistolica = st.slider('Informe sua pressao sistolica', min_value=0, max_value=2000)

#pressao arterial diastolica (número) (input de informações)
pressao_diastolica = st.slider('Informe sua pressao diastolica', min_value=0, max_value=2000)

#Colesterol (1-Normal, 2-Acima do normal, 3-Bem acima do normal) (input de informações)
opcoes = ['-','Normal','Acima do Normal','Bem acima do Normal']
colesterol = st.selectbox('Selecione a condição do seu colesterol', options=opcoes)

resultado = {'-':None,'Normal': 1, 'Acima do Normal': 2, 'Bem acima do Normal': 3}
colesterol_resultado = resultado.get(colesterol, None)

#Glicose (1-Normal, 2-Acima do normal, 3-Bem acima do normal)) (input de informações)
glicose = st.selectbox('Selecione a condição da sua Glicose', options=opcoes)
glicose_resultado = resultado.get(glicose, None)

#Fumante (sim,não) -(input de informações)
opcoes_2 = ['-','Sim','Não']
fumante = st.radio('Fumante?:', options = opcoes_2)

opcoes_s_n= {'-':None,'Sim': 1, 'Não': 2}
fumante_resultado = opcoes_s_n.get(fumante, None)

#Faz ingestão de alcool - (input de informações)
alcool = st.radio('Consome bebidas alcoolicas?:', options = opcoes_2)
alcool_resultado = opcoes_s_n.get(alcool, None)

#pratica atividade fisica - (input de informações)
at_fisica = st.radio('Pratica atividade fisita?:', options = opcoes_2)
at_fisica_resultado = opcoes_s_n.get(at_fisica, None)

#Criando a função para gerar o modelo de machine leanirg
@st.cache_data #função para armazenar em cache
def carregar_modelo_e_prever(valores, df):
    ml_model = RandomForestClassifier(n_estimators=20, n_jobs=4, max_depth=4)

    # Separando os dados em valor preditor e variáveis
    Y = df["cardio"]
    X = df.drop(["cardio", "id"], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)
    
    # Treine o modelo
    ml_model.fit(x_train, y_train)

    # Faça a previsão com base nos valores fornecidos
    resultado_modelo = ml_model.predict(valores)
    probabilidade = ml_model.predict_proba(valores)[:, 1][0]

    return resultado_modelo, probabilidade

df = st.session_state["data"]

#Salvando os dados preencidos pelo usuário
valores =np.array([[idade,sexo_resultado,peso,altura,pressao_sistolica,pressao_sistolica,colesterol_resultado,glicose_resultado,fumante_resultado,alcool_resultado,at_fisica_resultado]])

#Botão para efetuar previsão
botao = st.button('Efetuar previsão')

# Verifica se o modelo existe
if botao:
    if idade is not None and sexo_resultado is not None and peso is not None and altura is not None and pressao_sistolica is not None and pressao_diastolica is not None and colesterol_resultado is not None and glicose_resultado is not None and fumante_resultado is not None and alcool_resultado is not None and at_fisica_resultado is not None:
        with st.spinner('Carregando o modelo...'):

            # Chamando a função que carrega o modelo e faz a previsão
            resultado, probabilidade = carregar_modelo_e_prever(valores, df)

            # Exibir os resultados
            st.subheader('Modelo treinado!')
            probabilidade_texto = f'{probabilidade * 100:.2f}%'
            st.write(probabilidade_texto, ' de chance de ter doença cardíaca.')
        
        # Referência para o modelo considerado 70%
        referencia = 0.7  

        # Tomar uma decisão com base na probabilidade
        if probabilidade >= referencia:
            st.write('Recomendamos procurar um médico para avaliação')
            st.write('---------------------------------------------------------------------------')
            st.write('**Atenção!**')
            st.write('_Este modelo é apenas uma demonstração. Lembre-se de que é essencial buscar orientação médica para avaliação precisa de sua condição de saúde._')
        else:
            st.write('Probabilidade baixa de doença cardíaca. Consulte um médico para avaliação se necessário.')
            st.write('---------------------------------------------------------------------------')
            st.write('**Atenção!**')
            st.write('_Este modelo é apenas uma demonstração. Lembre-se de que é essencial buscar orientação médica para avaliação precisa de sua condição de saúde._')
    else:
        st.error("Todos os valores precisam ser preenchidos para efetuar a previsão.")


