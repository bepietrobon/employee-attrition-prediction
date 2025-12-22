# Predição de Rotatividade Voluntária com Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Dataset](https://img.shields.io/badge/Dataset-IBM%20HR%20Analytics-orange)](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
[![Institution](https://img.shields.io/badge/Institution-USP%2FECA-red)](https://www.eca.usp.br/)

Projeto desenvolvido para o **MBA em Business Intelligence & Analytics - USP/ECA.**  
Autora: Bettina Pietrobon Taucer Araujo

---

## 📋 Índice

- [Sobre o Projeto](#sobre-o-projeto)
- [Principais Resultados](#principais-resultados)
- [Metodologia](#metodologia)
- [Dataset](#dataset)
- [Instalação](#instalação)
- [Como Usar](#como-usar)
- [Modelo Final](#modelo-final)
- [Recomendações Estratégicas](#recomendações-estratégicas)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Publicações e Referências](#publicações-e-referências)
- [Contato](#contato)

---

## 🎓 Sobre o Projeto

Este projeto explora o uso da análise preditiva como mecanismo estratégico para reduzir a rotatividade de profissionais em empresas de tecnologia. Utilizando técnicas de **Machine Learning** aplicadas à base sintética **IBM HR Analytics Employee Attrition & Performance**, o estudo desenvolve modelos capazes de identificar colaboradores em risco de desligamento voluntário, permitindo ações preventivas de retenção.

A análise segue a metodologia **Knowledge Discovery in Databases (KDD)**, contemplando desde a exploração de dados até a construção e validação de modelos preditivos, com ênfase na interpretabilidade dos resultados para subsidiar decisões estratégicas de gestão de pessoas.

### 📍 Contexto

A alta rotatividade de profissionais em empresas de tecnologia impacta:

- **Custos** com recrutamento e treinamento
- **Produtividade** das equipes
- **Clima organizacional**
- **Capacidade de inovação**

### 🎯 Objetivo

Desenvolver um modelo preditivo que:

1. Identifique variáveis associadas ao desligamento voluntário
2. Antecipe comportamentos de saída com alta precisão
3. Oriente estratégias de retenção baseadas em dados

---

## 📑 Principais Resultados

### Modelo Final: Regressão Logística

| Métrica | Resultado |
|---------|-----------|
| **Recall** | **70,42%** |
| Acurácia | 77,32% |
| Precisão | 38,76% |
| F1 Score | 0,5000 |
| ROC AUC | 0,7732 |

**Critério de Seleção:** Modelo escolhido por apresentar o maior Recall e menor número de falsos negativos (21), priorizando a identificação de colaboradores em risco real de desligamento.

### Principais Fatores de Risco

1. **Tempo desde última promoção** (YearsSinceLastPromotion) - Estagnação na carreira
2. **Horas extras frequentes** (OverTime) - Sobrecarga de trabalho
3. **Cargo: Sales Representative** - Pressão por resultados
4. **Viagens frequentes** (BusinessTravel) - Instabilidade da rotina
5. **Alto número de empresas anteriores** - Perfil mais móvel

### Principais Fatores Protetivos

1. **Experiência profissional total** (TotalWorkingYears)
2. **Tempo na empresa** (YearsAtCompany)
3. **Satisfação no trabalho** (JobSatisfaction)
4. **Tempo com gestor atual** (YearsWithCurrManager)
5. **Envolvimento no trabalho** (JobInvolvement)

---

## 🧪 Metodologia

O projeto seguiu o processo **Knowledge Discovery in Databases (KDD)**:

### 1. Seleção dos Dados
- Dataset IBM HR Analytics (1.470 colaboradores)
- 35 variáveis (demográficas, profissionais, satisfação)

### 2. Pré-processamento
- Remoção de colunas irrelevantes
- OneHotEncoder para variáveis categóricas
- MinMaxScaler para normalização
- Balanceamento com class_weight='balanced'

### 3. Análise Exploratória (EDA)
- Distribuições e correlações
- Análise PCA (10 componentes, 90% variância)
- 17 visualizações geradas

### 4. Modelagem (Aprendizado Supervisionado)
- Regressão Logística
- Árvore de Decisão
- Random Forest
- Divisão: 70% treino / 30% teste

### 5. Avaliação e Interpretação
- Análise de métricas (Recall prioritizado)
- Interpretação de coeficientes
- Recomendações estratégicas

### Comparação de Modelos

| Modelo | Acurácia | Recall | F1 Score | Falsos Negativos |
|--------|----------|--------|----------|------------------|
| **Regressão Logística** | 77,32% | **70,42%** | 0,5000 | **21** |
| Árvore de Decisão | 74,83% | 23,94% | 0,2345 | 54 |
| Random Forest | **83,67%** | 11,27% | 0,1818 | 63 |

**Justificativa:** Embora o Random Forest tenha maior acurácia geral, a Regressão Logística identificou 70% dos colaboradores que realmente saíram, minimizando oportunidades perdidas de retenção.

---

## 💾 Dataset

### IBM HR Analytics Employee Attrition & Performance

- **Fonte:** [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- **Tipo:** Dataset sintético criado por cientistas da IBM
- **Tamanho:** 1.470 registros × 35 variáveis
- **Desbalanceamento:** 84% ativos vs. 16% desligados

### Categorias de Variáveis

**Demográficas**
- Age, Gender, MaritalStatus, Education, EducationField

**Profissionais**
- Department, JobRole, JobLevel, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion

**Satisfação**
- JobSatisfaction, EnvironmentSatisfaction, RelationshipSatisfaction, WorkLifeBalance

**Remuneração**
- MonthlyIncome, PercentSalaryHike, StockOptionLevel

**Comportamentais**
- OverTime, BusinessTravel, TrainingTimesLastYear

---

## 💻 Instalação

### Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/employee-attrition-prediction.git
cd employee-attrition-prediction

# Crie um ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Instale as dependências
pip install -r requirements.txt
```

---

## 🛠️ Como Usar

### Execução Básica

```bash
python pipeline_attrition_prediction.py
```

### Funcionamento do Script

O script executa automaticamente:

1. Download do dataset via KaggleHub
2. Análise exploratória completa com 17 visualizações
3. Pré-processamento e transformação dos dados
4. Treinamento de 3 modelos de Machine Learning
5. Avaliação e comparação de performance
6. Salvamento do modelo final e relatórios

### Estrutura de Saídas

```
project/
├── data/             # Dataset baixado
├── figures/          # 17 visualizações em alta resolução
├── models/           # Modelo final + preprocessador
└── reports/          # Métricas e resumo executivo
```

---

## 📈 Modelo Final

### Interpretação dos Coeficientes - Regressão Logística

#### Fatores de Risco (coeficientes positivos)

| Fator | Coeficiente | Interpretação |
|-------|-------------|---------------|
| YearsSinceLastPromotion | +1.89 | Estagnação na carreira aumenta risco |
| OverTime_Yes | +1.67 | Horas extras frequentes são críticas |
| JobRole_Sales Representative | +1.43 | Pressão comercial elevada |
| NumCompaniesWorked | +1.28 | Histórico de mobilidade |
| BusinessTravel_Travel_Frequently | +1.15 | Instabilidade na rotina |

#### Fatores Protetivos (coeficientes negativos)

| Fator | Coeficiente | Interpretação |
|-------|-------------|---------------|
| TotalWorkingYears | -1.52 | Experiência retém talentos |
| YearsAtCompany | -1.38 | Vínculo com empresa protege |
| JobSatisfaction | -1.21 | Satisfação é barreira ao turnover |
| YearsWithCurrManager | -1.09 | Relação com gestor importa |
| JobInvolvement | -0.94 | Engajamento previne saída |

### Matriz de Confusão - Modelo Final

**Performance no conjunto de teste (441 registros):**

|  | **Predito: Ativo** | **Predito: Desligado** | **Total** |
|---|:---:|:---:|:---:|
| **Real: Ativo** | 291 | 79 | 370 |
| **Real: Desligado** | 21 | 50 | 71 |
| **Total** | 312 | 129 | 441 |

**Interpretação dos Resultados:**

- **Verdadeiros Negativos (291):** Colaboradores ativos corretamente identificados como permanecendo na empresa
- **Verdadeiros Positivos (50):** Colaboradores em risco corretamente identificados → **70,4% de recall**
- **Falsos Negativos (21):** Colaboradores que saíram mas não foram identificados → **29,6% não detectados**
- **Falsos Positivos (79):** Alertas para colaboradores que permaneceram → **21,4% de falsos alarmes**

**Por que este modelo foi escolhido:** Os 21 falsos negativos representam o menor número entre os 3 modelos testados, maximizando as oportunidades de retenção.

---

## ✨ Recomendações Estratégicas

### 1. Políticas de Progressão na Carreira

**Problema:** Tempo desde última promoção é o fator de maior risco.

**Ações:**
- Implementar ciclos de avaliação e promoção mais frequentes
- Criar trilhas de carreira claras e transparentes
- Estabelecer PDI (Plano de Desenvolvimento Individual) para todos
- Comunicar critérios de promoção de forma objetiva

### 2. Gestão de Carga de Trabalho

**Problema:** Horas extras frequentes aumentam significativamente o risco.

**Ações:**
- Monitorar horas extras por colaborador/equipe
- Estabelecer limites e compensações adequadas
- Avaliar dimensionamento de equipes
- Implementar ferramentas de gestão de tempo

### 3. Fortalecimento da Liderança

**Problema:** Tempo com gestor atual é fator protetivo importante.

**Ações:**
- Capacitar líderes em gestão de pessoas
- Implementar 1-on-1s regulares
- Criar cultura de feedback contínuo
- Avaliar clima das equipes periodicamente

### 4. Programas de Engajamento

**Problema:** Satisfação e envolvimento protegem contra saída.

**Ações:**
- Pesquisas de clima trimestrais
- Programas de reconhecimento
- Oportunidades de desenvolvimento
- Projetos desafiadores e significativos

### 5. Atenção a Cargos e Rotinas Específicas

**Problema:** Sales Representatives e viagens frequentes elevam risco.

**Ações:**
- Planos de carreira diferenciados para vendas
- Revisão de políticas de viagem
- Suporte adicional para cargos de alta pressão
- Benefícios compensatórios

---

## 🌐 Tecnologias Utilizadas

### Core
- Python 3.8+
- Google Colab

### Análise de Dados
- Pandas 2.0+
- NumPy 1.24+
- Scikit-learn 1.3+

### Visualização
- Matplotlib 3.7+
- Seaborn 0.12+

### Utilitários
- KaggleHub (download automatizado)
- Joblib (persistência de modelos)

### Instalação

```bash
pip install -r requirements.txt
```

---

## 📖 Publicações e Referências

**Título:** Predição de Rotatividade Voluntária com Machine Learning: Caso Aplicado à Base IBM HR Analytics

**Autora:** Bettina Pietrobon Taucer Araujo  
**Instituição:** Universidade de São Paulo (USP) - Escola de Comunicações e Artes (ECA)    
**Orientador**: Prof. Paulo Henrique Assis Feitosa  
**Ano**: 2025

### Principais Referências

1. CHIAVENATO, I. Gestão de pessoas: o novo papel dos recursos humanos nas organizações. 4. ed. Rio de Janeiro: Elsevier, 2014.

2. PUNNOOSE, R.; AJIT, P. Prediction of employee turnover in organizations using Machine Learning algorithms. International Journal of Advanced Research in Artificial Intelligence, 2016.

3. GÉRON, A. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow. 2. ed. O'Reilly Media, 2019.

4. BEN YAHIA, N.; HLEL, J.; COLOMO-PALACIOS, R. From Big Data to Deep Data to Support People Analytics for Employee Attrition Prediction. IEEE Access, 2021.

5. MITCHELL, T. M. Machine Learning. New York: McGraw-Hill, 1997.

---

## 🗺️ Áreas para Contribuição

- Otimização de hiperparâmetros
- Novas visualizações
- Modelos adicionais (XGBoost, LightGBM)
- Interface web interativa
- Tradução da documentação
- Testes unitários

---

## 📧 Contato

**LinkedIn:** [linkedin.com/in/bettina-araujo](https://linkedin.com/in/bettinataraujo)

---

Este projeto está licenciado sob a Licença MIT. Consulte o arquivo [LICENSE](LICENSE) para detalhes.

Se este projeto foi útil, considere dar uma ⭐ no repositório!


**[⬆ Voltar ao topo](#predição-de-rotatividade-voluntária-com-machine-learning)**
