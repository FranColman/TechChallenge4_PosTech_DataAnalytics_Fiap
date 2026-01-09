# TechChallenge4_PosTech_DataAnalytics_Fiap

🏥 Sistema Preditivo e Analítico de Obesidade

Tech Challenge – Fase 4 | Pós-Graduação em Data Analytics

📌 Visão Geral

Este projeto apresenta o desenvolvimento de um sistema preditivo de obesidade integrado a um dashboard analítico, com foco em aplicar, de forma prática, os conceitos aprendidos na Fase 4 do curso de Data Analytics.

A solução combina Machine Learning, engenharia de dados, avaliação de modelos e visualização analítica, entregando uma aplicação funcional que simula um cenário real de uso em apoio à tomada de decisão.

⚠️ Aviso importante
Este sistema possui finalidade acadêmica e analítica, servindo como apoio à decisão. Ele não substitui avaliação médica ou profissional especializada.

🎯 Objetivo do Projeto

O projeto possui dois objetivos principais:

Sistema Preditivo
Estimar o nível de obesidade de um indivíduo com base em dados demográficos, físicos e comportamentais, utilizando um modelo de Machine Learning.

Dashboard Analítico
Fornecer uma visão exploratória e estratégica dos dados, permitindo identificar padrões, perfis e relações relevantes para análise de negócio.

👥 Autores

Franco Colmán

Hugo Duran

Projeto desenvolvido como parte do Tech Challenge da Fase 4, com foco em colocar em prática os conceitos estudados na pós-graduação em Data Analytics, integrando modelagem estatística, aprendizado de máquina e visualização de dados.

🧠 Estratégia de Machine Learning
📊 Base de Dados

Utilizamos o Obesity Dataset, que contém informações sobre:

Dados demográficos: Age, Gender

Medidas físicas: Height, Weight

Hábitos alimentares e estilo de vida:
FCVC, NCP, CH2O, FAF, TUE, FAVC, CAEC, CALC

Histórico e comportamento:
family_history, SMOKE, SCC, MTRANS

Variável alvo: Obesity (7 classes)

O problema é modelado como uma classificação multiclasse, conforme explorado ao longo da Fase 4.

⚙️ Modelo Utilizado

O algoritmo escolhido foi o Gradient Boosting, por sua capacidade de:

Capturar relações não lineares

Trabalhar bem com dados tabulares

Modelar interações complexas entre variáveis

Apresentar alto desempenho em problemas reais de classificação

O modelo foi implementado dentro de um pipeline, contendo:

Separação de variáveis numéricas e categóricas

One-Hot Encoding para variáveis categóricas

Padronização das variáveis numéricas

Integração completa do pré-processamento ao modelo

Essa abordagem garante consistência, reprodutibilidade e segurança entre treinamento e inferência.

📈 Avaliação do Modelo

O desempenho foi avaliado utilizando:

Acurácia

Precisão

Recall

F1-score por classe

O modelo apresentou acurácia aproximada de 95%, com desempenho consistente entre as classes, demonstrando boa capacidade de generalização.

🖥️ Sistema Preditivo (Aplicação)

A aplicação foi desenvolvida com Streamlit, oferecendo uma interface simples e intuitiva.

Funcionalidades principais:

Entrada de dados do paciente (interface em português)

Conversão automática para o padrão do pipeline (inglês)

Cálculo automático do IMC

Predição do nível de obesidade

Exibição das probabilidades por classe, aumentando a transparência do modelo

📊 Dashboard Analítico

Além da predição individual, o sistema inclui uma área de análise exploratória, com gráficos voltados à visão de negócio.

Principais análises:

Distribuição dos níveis de obesidade (contagem e percentual)

Dispersão de peso × altura por classe

Composição de obesidade por gênero (100% empilhado)

Heatmap de faixa etária × obesidade

Heatmap de correlação entre variáveis numéricas

Gráfico radar com perfil médio de hábitos por nível de obesidade

Essas análises permitem identificar padrões populacionais, clusters e relações relevantes para tomada de decisão.

🚀 Deploy

A aplicação foi:

Versionada com Git e GitHub

Publicada no Streamlit Cloud

Configurada com requirements.txt e runtime.txt

Disponibilizada para acesso remoto

Isso garante reprodutibilidade, portabilidade e aderência a boas práticas de entrega.

🎓 Conclusão

Este projeto consolida os aprendizados da Fase 4 da pós-graduação em Data Analytics, integrando:

Engenharia de dados

Machine Learning com Gradient Boosting

Avaliação de modelos

LINK - STREAMLIT: https://techchallenge4postechdataanalyticsfiap-jcwkcts8n92sjk8vkqt8ds.streamlit.app/

LINK - GITHUB: https://github.com/FranColman/TechChallenge4_PosTech_DataAnalytics_Fiap

Visualização analítica

Deploy de aplicações de dados

A solução simula um cenário real de Data Analytics, indo além do modelo isolado e entregando valor de negócio por meio de um sistema completo.
