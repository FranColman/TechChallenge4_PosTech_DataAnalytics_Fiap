# 🏥 Sistema Preditivo e Analítico de Obesidade

**Tech Challenge – Fase 4 | Pós-Graduação em Data Analytics**

---

## 📌 Visão Geral

Este projeto apresenta o desenvolvimento de um **sistema preditivo de obesidade integrado a um dashboard analítico**, com foco em **apoio à tomada de decisão**.

A solução foi construída como parte do **Tech Challenge – Fase 4**, com o objetivo de **colocar em prática os conceitos aprendidos ao longo da pós-graduação em Data Analytics**, integrando:

- Análise exploratória de dados  
- Modelagem estatística  
- Aprendizado de Máquina  
- Visualização de dados orientada ao negócio  

O sistema permite tanto a **predição do nível de obesidade de um indivíduo**, quanto a **análise de padrões populacionais** relacionados a hábitos, perfil físico e histórico familiar.

---

## 🎯 Objetivo do Projeto

O principal objetivo é demonstrar, de forma prática e aplicada, como técnicas de **Machine Learning** e **Analytics** podem ser utilizadas para:

- Estimar o nível de obesidade com base em características individuais
- Identificar padrões relevantes no conjunto de dados
- Traduzir informações técnicas em **insights claros para o negócio**

O sistema **não substitui avaliação clínica**, sendo uma ferramenta de **apoio analítico e educacional**.

---

## 🧠 Modelo de Machine Learning

O modelo utilizado neste projeto é o **Gradient Boosting Classifier**, escolhido por apresentar:

- Boa performance em dados tabulares
- Capacidade de capturar relações não lineares
- Robustez frente a variáveis heterogêneas (numéricas e categóricas)

---

### 🔍 Pipeline do Modelo

O pipeline de Machine Learning inclui:

- **Pré-processamento**
  - Padronização de variáveis numéricas
  - Codificação de variáveis categóricas
- **Treinamento**
  - Modelo: `GradientBoostingClassifier`
- **Avaliação**
  - Métrica principal: **Acurácia**
  - Resultado obtido: **95% de acurácia**

O pipeline completo foi serializado e salvo no arquivo:

---

## 📊 Dashboard Analítico

Além da predição individual, o projeto conta com um **dashboard analítico interativo**, que permite explorar o comportamento da base de dados sob uma ótica de negócio.

### Principais análises disponíveis:

- Distribuição dos níveis de obesidade (contagem e percentual)
- Relação entre peso e altura por nível de obesidade
- Composição de obesidade por gênero (100% empilhado)
- Distribuição de níveis por faixa etária (heatmap)
- Correlação entre variáveis numéricas
- Perfil médio de hábitos por nível de obesidade (radar normalizado)

Essas visualizações ajudam a responder perguntas como:
- Onde estão concentrados os maiores riscos?
- Como hábitos impactam os níveis de obesidade?
- Existem diferenças relevantes por gênero ou idade?

---

## 🖥️ Aplicação

A aplicação foi desenvolvida em **Streamlit**, com foco em:

- Interface limpa e intuitiva
- Boa usabilidade
- Navegação clara entre predição e análises

### Funcionalidades principais:

- Formulário de predição individual
- Cálculo automático de IMC
- Exibição da classe prevista
- Visualização de métricas e gráficos analíticos
- Download do dataset

---

## ☁️ Deploy

O projeto é totalmente compatível com **Streamlit Cloud**, permitindo que a aplicação seja disponibilizada publicamente de forma simples.

### Requisitos para deploy

- Repositório versionado no **GitHub**
- Arquivo `requirements.txt` com todas as dependências do projeto
- Arquivo `runtime.txt` especificando a versão do Python utilizada
- Modelo treinado (`obesity_pipeline.joblib`) incluído no repositório

---

## 👥 Autores

Projeto desenvolvido por:

- **Franco Colmán**
- **Hugo Duran**

Como parte da **Pós-Graduação em Data Analytics**, com foco na **aplicação prática dos conceitos estudados na Fase 4 do curso**.

---

## 📌 Considerações Finais

Este projeto consolida conhecimentos técnicos e analíticos em uma **solução completa de Data Analytics**, cobrindo todo o ciclo:

- Análise exploratória de dados  
- Modelagem estatística e Machine Learning  
- Construção de um sistema preditivo  
- Desenvolvimento de um dashboard analítico orientado ao negócio  

A solução demonstra como dados podem ser transformados em **insights acionáveis**, reforçando a importância da **visão de negócio aliada à modelagem e visualização de dados**.

 ## 🔗 Links do Projeto

- 🌐 **Aplicação no Streamlit Cloud:**  
  [Acessar aplicação](https://techchallenge4postechdataanalyticsfiap-jcwkcts8n92sjk8vkqt8ds.streamlit.app/)

- 💻 **Repositório no GitHub:**  
  [Acessar repositório](https://github.com/FranColman/TechChallenge4_PosTech_DataAnalytics_Fiap)
