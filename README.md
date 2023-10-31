**Introdução**
O modelo da aplicação foi dividido em 3 páginas:

Página 1: Home
Nesta página, realizado o processamento e tratamento dos dados, além de incluir uma descrição do projeto.

Página 2: Previsão
Disponibilizado as variáveis para que o usuário insira os valores. pós preenchê-los, o algoritmo de Machine Learning é acionado.

Página 3: Resultados do Modelo
Aqui, incluído as métricas do algoritmo para que a performance possa ser avaliada.

**Deploy**
Para fazer o deploy foi utilizado o servido (Streamlit) <https://streamlit.io/>, neste servidor é possivel disponibilizar até 1 projeto de forma gratuita.

Abaixo, segue o passo a passo para fazer o deploy do modelo.
- Criar a conta em (Streamlit) <https://streamlit.io/>
- Subir o projeto para um repositório do GitHub
- Criar a conexão do Streamlit no Github
- Executar o deploy do modelo


**Execução local**
Se desejar rodar o modelo em um ambiente local, siga as etapas abaixo:

- Clone o repositório do GitHub.
- No terminal de comando, navegue até a pasta onde está o repositório clonado.
- Execute o seguinte comando: streamlit run 'nome do arquivo'.

Exemplo: streamlit run 1_home.py

