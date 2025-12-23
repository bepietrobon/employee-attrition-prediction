"""
Pipeline para Predição de Rotatividade Voluntária (Employee Attrition)
==================================================================================

Especialização em Business Intelligence & Analytics - USP/ECA
Autora: Bettina Pietrobon Taucer Araujo
Orientador: Prof. Paulo Henrique Assis Feitosa

Dataset: IBM HR Analytics Employee Attrition & Performance
Disponível em: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

Descrição:
----------
Este pipeline implementa um estudo completo de análise preditiva para identificar
colaboradores em risco de desligamento voluntário, utilizando técnicas de Machine
Learning. O modelo final (Regressão Logística) foi escolhido por apresentar o melhor
equilíbrio entre métricas, com destaque para recall (70,42%) e menor número de
falsos negativos, fatores críticos para estratégias de retenção.

Principais Achados:
-------------------
- Fatores de RISCO: Tempo desde última promoção, horas extras frequentes, 
  cargo de Sales Representative, viagens frequentes
- Fatores PROTETIVOS: Experiência profissional, tempo de empresa, satisfação
  no trabalho, relacionamento com gestor

Compatível com: Google Colab e execução local
Linguagem: Python 3.8+
"""

# =============================================================================
# IMPORTAÇÃO DE BIBLIOTECAS
# =============================================================================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import kagglehub
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score
)
from sklearn.utils.class_weight import compute_class_weight

import joblib

# =============================================================================
# CONFIGURAÇÕES GLOBAIS
# =============================================================================
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Cores padronizadas
COLORS = {
    'active': '#2ecc71',      # Verde para colaboradores ativos
    'attrition': '#e74c3c',   # Vermelho para desligados
    'positive': '#e74c3c',    # Fatores de risco
    'negative': '#2ecc71',    # Fatores protetivos
    'neutral': '#3498db'      # Neutro
}

# Diretórios
DATA_DIR = "data"
REPORTS_DIR = "reports"
FIGURES_DIR = "figures"

# Criar estrutura de diretórios
for directory in [DATA_DIR, REPORTS_DIR, FIGURES_DIR]:
    os.makedirs(directory, exist_ok=True)


# =============================================================================
# FUNÇÃO: DOWNLOAD DO DATASET
# =============================================================================
def download_dataset():
    """
    Faz download automático do dataset IBM HR Analytics via KaggleHub.
    
    Funciona automaticamente no Google Colab sem configuração.
    Para execução local, requer arquivo kaggle.json configurado em ~/.kaggle/
    
    Returns:
        str: Caminho para o arquivo CSV baixado
    """
    print("\n" + "="*70)
    print("📥 DOWNLOAD DO DATASET")
    print("="*70)
    
    try:
        print("🔄 Baixando dataset do Kaggle via KaggleHub...")
        print("   Dataset: IBM HR Analytics Employee Attrition & Performance")
        
        # Download do dataset
        path = kagglehub.dataset_download("pavansubhasht/ibm-hr-analytics-attrition-dataset")
        
        print(f"✅ Download concluído!")
        print(f"📂 Diretório do dataset: {path}")
        
        # Procurar o arquivo CSV
        csv_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.csv'):
                    csv_files.append(os.path.join(root, file))
        
        if not csv_files:
            raise FileNotFoundError("Nenhum arquivo CSV encontrado no dataset baixado")
        
        dataset_path = csv_files[0]
        print(f"📄 Arquivo encontrado: {os.path.basename(dataset_path)}")
        
        # Copiar para nossa pasta /data
        import shutil
        local_path = os.path.join(DATA_DIR, "IBM_HR_Analytics.csv")
        shutil.copy2(dataset_path, local_path)
        print(f"💾 Cópia salva em: {local_path}")
        
        return local_path
    
    except Exception as e:
        print(f"\n❌ Erro ao baixar dataset: {str(e)}")
        print("\n💡 Dica: Para uso local, configure seu arquivo kaggle.json")
        print("   Veja: https://www.kaggle.com/docs/api")
        raise


# =============================================================================
# FUNÇÃO: CARREGAR DADOS
# =============================================================================
def load_data(path):
    """
    Carrega o dataset do IBM HR Analytics.
    
    Args:
        path (str): Caminho para o arquivo CSV
        
    Returns:
        pd.DataFrame: Dataset carregado
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    
    df = pd.read_csv(path)
    
    print("\n" + "="*70)
    print("📂 DATASET CARREGADO")
    print("="*70)
    print(f"   Dimensões: {df.shape[0]} linhas × {df.shape[1]} colunas")
    
    if 'Attrition' in df.columns:
        attrition_counts = df['Attrition'].value_counts().to_dict()
        print(f"\n   📊 Distribuição de Attrition:")
        total = sum(attrition_counts.values())
        for key, value in attrition_counts.items():
            pct = (value / total) * 100
            print(f"      • {key}: {value} ({pct:.2f}%)")
    
    return df


# =============================================================================
# FUNÇÃO: ANÁLISE EXPLORATÓRIA DE DADOS (EDA)
# =============================================================================
def exploratory_analysis_complete(df):
    """
    Gera análise exploratória COMPLETA com múltiplos gráficos conforme TCC.
    
    Gráficos gerados:
    1. Overview geral (idade, estado civil, tempo de empresa, tempo no cargo)
    2. Distribuição de Attrition (pizza + barras)
    3. Análise por Departamento e Cargo
    4. Fatores de Satisfação (4 dimensões)
    5. Análise Salarial (distribuição e por nível)
    6. Horas Extras e Viagens
    """
    print("\n" + "="*70)
    print("📊 ANÁLISE EXPLORATÓRIA DETALHADA (EDA)")
    print("="*70)
    
    df_plot = df.copy()
    df_plot["Attrition"] = df_plot["Attrition"].map({
        1: "Desligado", 0: "Ativo", 
        "Yes": "Desligado", "No": "Ativo"
    })
    palette = {"Ativo": COLORS['active'], "Desligado": COLORS['attrition']}
    
    # =========================================================================
    # 1. OVERVIEW GERAL
    # =========================================================================
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    sns.histplot(data=df_plot, x="Age", hue="Attrition", multiple="stack",
                 palette=palette, ax=axs[0, 0], kde=True)
    axs[0, 0].set_title("Distribuição por Idade", fontsize=12, fontweight='bold')
    axs[0, 0].set_xlabel("Idade")
    
    sns.countplot(data=df_plot, x="MaritalStatus", hue="Attrition",
                  palette=palette, ax=axs[0, 1])
    axs[0, 1].set_title("Attrition por Estado Civil", fontsize=12, fontweight='bold')
    axs[0, 1].set_xlabel("Estado Civil")
    
    sns.histplot(data=df_plot, x="YearsAtCompany", hue="Attrition", multiple="stack",
                 palette=palette, ax=axs[1, 0], kde=True)
    axs[1, 0].set_title("Tempo de Empresa", fontsize=12, fontweight='bold')
    axs[1, 0].set_xlabel("Anos na Empresa")
    
    sns.histplot(data=df_plot, x="YearsInCurrentRole", hue="Attrition", multiple="stack",
                 palette=palette, ax=axs[1, 1], kde=True)
    axs[1, 1].set_title("Tempo no Cargo Atual", fontsize=12, fontweight='bold')
    axs[1, 1].set_xlabel("Anos no Cargo Atual")
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/01_eda_overview.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Figura 1: Overview geral")
    
    # =========================================================================
    # 2. DISTRIBUIÇÃO DE ATTRITION
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Gráfico de pizza
    attrition_counts = df_plot['Attrition'].value_counts()
    colors = [palette['Ativo'], palette['Desligado']]
    ax1.pie(attrition_counts.values, labels=attrition_counts.index, autopct='%1.1f%%',
            startangle=90, colors=colors, textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax1.set_title('Distribuição de Attrition', fontsize=14, fontweight='bold')
    
    # Gráfico de barras
    sns.countplot(data=df_plot, x='Attrition', palette=colors, ax=ax2)
    ax2.set_title('Contagem de Attrition', fontsize=14, fontweight='bold')
    for container in ax2.containers:
        ax2.bar_label(container, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/02_attrition_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Figura 2: Distribuição de Attrition")
    
    # =========================================================================
    # 3. ANÁLISE POR DEPARTAMENTO E CARGO
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Por Departamento
    dept_attrition = pd.crosstab(df_plot['Department'], df_plot['Attrition'], 
                                  normalize='index') * 100
    dept_attrition.plot(kind='bar', stacked=True, color=colors, ax=ax1)
    ax1.set_title('Attrition por Departamento (%)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Departamento')
    ax1.set_ylabel('Percentual (%)')
    ax1.legend(title='Status')
    ax1.tick_params(axis='x', rotation=45)
    
    # Por JobRole (top 10)
    role_attrition = df_plot.groupby('JobRole')['Attrition'].apply(
        lambda x: (x == 'Desligado').sum() / len(x) * 100
    ).sort_values(ascending=False).head(10)
    
    role_attrition.plot(kind='barh', color=COLORS['attrition'], ax=ax2)
    ax2.set_title('Taxa de Attrition por Cargo (Top 10)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Taxa de Attrition (%)')
    ax2.set_ylabel('Cargo')
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/03_dept_role_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Figura 3: Análise por Departamento e Cargo")
    
    # =========================================================================
    # 4. FATORES DE SATISFAÇÃO
    # =========================================================================
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    satisfaction_cols = ['JobSatisfaction', 'EnvironmentSatisfaction',
                        'RelationshipSatisfaction', 'WorkLifeBalance']
    titles = ['Satisfação no Trabalho', 'Satisfação com Ambiente',
              'Satisfação com Relacionamentos', 'Equilíbrio Vida-Trabalho']
    
    for idx, (col, title) in enumerate(zip(satisfaction_cols, titles)):
        row, col_idx = idx // 2, idx % 2
        sns.countplot(data=df_plot, x=col, hue='Attrition', palette=palette, 
                     ax=axs[row, col_idx])
        axs[row, col_idx].set_title(title, fontsize=11, fontweight='bold')
        axs[row, col_idx].legend(title='Status')
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/04_satisfaction_factors.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Figura 4: Fatores de Satisfação")
    
    # =========================================================================
    # 5. ANÁLISE SALARIAL
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Distribuição de salário por attrition
    sns.violinplot(data=df_plot, x='Attrition', y='MonthlyIncome', 
                   palette=colors, ax=ax1)
    ax1.set_title('Distribuição de Salário por Status', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Salário Mensal (USD)')
    
    # BoxPlot de salário por JobLevel e Attrition
    sns.boxplot(data=df_plot, x='JobLevel', y='MonthlyIncome', hue='Attrition',
                palette=palette, ax=ax2)
    ax2.set_title('Salário por Nível e Status', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Salário Mensal (USD)')
    ax2.legend(title='Status')
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/05_salary_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Figura 5: Análise Salarial")
    
    # =========================================================================
    # 6. HORAS EXTRAS E VIAGENS
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Horas extras
    overtime_attrition = pd.crosstab(df_plot['OverTime'], df_plot['Attrition'], 
                                      normalize='index') * 100
    overtime_attrition.plot(kind='bar', color=colors, ax=ax1)
    ax1.set_title('Impacto de Horas Extras no Attrition', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Faz Hora Extra?')
    ax1.set_ylabel('Percentual (%)')
    ax1.legend(title='Status')
    ax1.tick_params(axis='x', rotation=0)
    
    # Viagens
    travel_attrition = pd.crosstab(df_plot['BusinessTravel'], df_plot['Attrition'], 
                                    normalize='index') * 100
    travel_attrition.plot(kind='bar', color=colors, ax=ax2)
    ax2.set_title('Impacto de Viagens a Trabalho no Attrition', 
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('Frequência de Viagens')
    ax2.set_ylabel('Percentual (%)')
    ax2.legend(title='Status')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/06_overtime_travel.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Figura 6: Horas Extras e Viagens")


# =============================================================================
# FUNÇÃO: MATRIZ DE CORRELAÇÃO
# =============================================================================
def plot_correlation_matrix(df):
    """
    Gera matriz de correlação das variáveis numéricas.
    
    Produz dois gráficos:
    1. Matriz de correlação completa
    2. Top 15 variáveis correlacionadas com Attrition
    """
    print("\n" + "="*70)
    print("🔗 MATRIZ DE CORRELAÇÃO")
    print("="*70)
    
    # Mapear Attrition para numérico se necessário
    df_corr = df.copy()
    if df_corr['Attrition'].dtype == 'object':
        df_corr['Attrition'] = df_corr['Attrition'].map({'Yes': 1, 'No': 0})
    
    # Selecionar apenas variáveis numéricas
    numeric_cols = df_corr.select_dtypes(include=[np.number]).columns.tolist()
    
    # Calcular correlação
    corr_matrix = df_corr[numeric_cols].corr()
    
    # Plotar matriz completa
    plt.figure(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdBu_r',
                center=0, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8})
    plt.title('Matriz de Correlação - Variáveis Numéricas',
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/07_correlation_matrix_full.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Figura 7: Matriz de Correlação Completa")
    
    # Correlação com Attrition (TOP 15)
    attrition_corr = corr_matrix['Attrition'].abs().sort_values(ascending=False)[1:16]
    
    plt.figure(figsize=(10, 8))
    colors_corr = [COLORS['attrition'] if x > 0 else COLORS['active']
                   for x in corr_matrix['Attrition'][attrition_corr.index]]
    attrition_corr.plot(kind='barh', color=colors_corr)
    plt.title('Top 15 Variáveis Correlacionadas com Attrition',
              fontsize=12, fontweight='bold')
    plt.xlabel('Correlação Absoluta')
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/08_attrition_correlation.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Figura 8: Correlação com Attrition (Top 15)")


# =============================================================================
# FUNÇÃO: PRÉ-PROCESSAMENTO
# =============================================================================
def preprocess(df):
    """
    Aplica pipeline de pré-processamento conforme metodologia do TCC.
    
    Etapas:
    1. Remove colunas sem valor preditivo
    2. Mapeia variável-alvo para numérico
    3. Identifica variáveis categóricas e numéricas
    4. Aplica OneHotEncoder e MinMaxScaler
    
    Returns:
        tuple: (X_scaled, y, preprocessor, feature_names)
    """
    print("\n" + "="*70)
    print("🔧 PRÉ-PROCESSAMENTO")
    print("="*70)
    
    df = df.copy()
    
    # Remover colunas sem valor preditivo
    cols_to_drop = ["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"]
    existing_cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    
    if existing_cols_to_drop:
        df.drop(columns=existing_cols_to_drop, inplace=True)
        print(f"   Colunas removidas: {existing_cols_to_drop}")
    
    # Mapear variável-alvo
    df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})
    
    y = df["Attrition"]
    X = df.drop(columns=["Attrition"])
    
    # Identificar tipos de variáveis
    categorical = X.select_dtypes(include="object").columns.tolist()
    numerical = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    
    print(f"   Variáveis categóricas: {len(categorical)}")
    print(f"   Variáveis numéricas: {len(numerical)}")
    
    # Configurar pipeline de transformação
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", sparse_output=False, 
                                 handle_unknown="ignore"), categorical),
            ("num", MinMaxScaler(), numerical)
        ],
        remainder="drop"
    )
    
    # Aplicar transformação
    X_transformed = preprocessor.fit_transform(X)
    
    # Criar nomes de colunas
    ohe = preprocessor.named_transformers_["cat"]
    cat_names = ohe.get_feature_names_out(categorical).tolist()
    feature_names = cat_names + numerical
    
    X_scaled = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)
    
    print(f"✅ Pré-processamento concluído: {X_scaled.shape[1]} features geradas")
    
    return X_scaled, y, preprocessor, feature_names


# =============================================================================
# FUNÇÃO: ANÁLISE PCA COMPLETA
# =============================================================================
def run_pca_complete(X_scaled, y, n_components=10):
    """
    Executa análise PCA COMPLETA conforme TCC.
    
    Gráficos gerados:
    1. Scree Plot (variância explicada)
    2. Biplot (PC1 vs PC2)
    3. Loadings (contribuição das variáveis)
    4. Gráfico 3D (PC1, PC2, PC3)
    
    Returns:
        tuple: (pca, scores)
    """
    print("\n" + "="*70)
    print("🔍 ANÁLISE DE COMPONENTES PRINCIPAIS (PCA) - COMPLETA")
    print("="*70)
    
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_scaled)
    
    # =========================================================================
    # 1. SCREE PLOT
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Variância individual
    ax1.bar(range(1, n_components + 1), pca.explained_variance_ratio_,
            alpha=0.7, color='steelblue')
    ax1.set_title("Variância Explicada por Componente", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Componente Principal")
    ax1.set_ylabel("% Variância Explicada")
    ax1.grid(True, alpha=0.3)
    
    # Variância acumulada
    cumsum = pca.explained_variance_ratio_.cumsum()
    ax2.plot(range(1, n_components + 1), cumsum, marker="o",
             linewidth=2, markersize=8, color='darkgreen')
    ax2.axhline(y=0.8, color='r', linestyle='--', label='80% variância')
    ax2.axhline(y=0.9, color='orange', linestyle='--', label='90% variância')
    ax2.set_title("Variância Acumulada", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Número de Componentes")
    ax2.set_ylabel("% Variância Acumulada")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/09_pca_scree_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Variância explicada: {cumsum[-1]*100:.2f}% com {n_components} componentes")
    print("   ✅ Figura 9: PCA Scree Plot")
    
    # =========================================================================
    # 2. BIPLOT
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plotar scores
    colors_map = {0: COLORS['active'], 1: COLORS['attrition']}
    for attrition_val in [0, 1]:
        mask = y == attrition_val
        label = 'Ativo' if attrition_val == 0 else 'Desligado'
        ax.scatter(scores[mask, 0], scores[mask, 1],
                  c=colors_map[attrition_val], label=label,
                  alpha=0.6, s=50)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                  fontsize=12, fontweight='bold')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
                  fontsize=12, fontweight='bold')
    ax.set_title('PCA Biplot - PC1 vs PC2', fontsize=14, fontweight='bold')
    ax.legend(title='Status', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/10_pca_biplot.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Figura 10: PCA Biplot")
    
    # =========================================================================
    # 3. LOADINGS
    # =========================================================================
    loadings = pca.components_
    feature_names = X_scaled.columns
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # PC1
    pc1_loadings = pd.Series(loadings[0], index=feature_names)
    top_pc1 = pc1_loadings.abs().nlargest(10)
    colors_pc1 = [COLORS['attrition'] if pc1_loadings[idx] > 0 
                  else COLORS['active'] for idx in top_pc1.index]
    pc1_loadings[top_pc1.index].sort_values().plot(kind='barh', color=colors_pc1, ax=ax1)
    ax1.set_title(f'Top 10 Features - PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                  fontsize=11, fontweight='bold')
    ax1.set_xlabel('Loading')
    
    # PC2
    pc2_loadings = pd.Series(loadings[1], index=feature_names)
    top_pc2 = pc2_loadings.abs().nlargest(10)
    colors_pc2 = [COLORS['attrition'] if pc2_loadings[idx] > 0 
                  else COLORS['active'] for idx in top_pc2.index]
    pc2_loadings[top_pc2.index].sort_values().plot(kind='barh', color=colors_pc2, ax=ax2)
    ax2.set_title(f'Top 10 Features - PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
                  fontsize=11, fontweight='bold')
    ax2.set_xlabel('Loading')
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/11_pca_loadings.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Figura 11: PCA Loadings")
    
    # =========================================================================
    # 4. GRÁFICO 3D
    # =========================================================================
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    for attrition_val in [0, 1]:
        mask = y == attrition_val
        label = 'Ativo' if attrition_val == 0 else 'Desligado'
        ax.scatter(scores[mask, 0], scores[mask, 1], scores[mask, 2],
                  c=colors_map[attrition_val], label=label,
                  alpha=0.6, s=30)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', 
                  fontweight='bold')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', 
                  fontweight='bold')
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)', 
                  fontweight='bold')
    ax.set_title('PCA 3D - Três Primeiros Componentes', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(title='Status')
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/12_pca_3d.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Figura 12: PCA 3D")
    
    return pca, scores


# =============================================================================
# FUNÇÃO: TREINAMENTO DOS MODELOS
# =============================================================================
def train_models(X, y):
    """
    Treina e avalia múltiplos modelos de classificação.
    
    Modelos testados:
    - Regressão Logística (MODELO FINAL escolhido)
    - Árvore de Decisão
    - Random Forest
    
    Returns:
        tuple: (modelos, resultados, matrizes, predictions, probabilities, X_test, y_test)
    """
    print("\n" + "="*70)
    print("🤖 TREINAMENTO E AVALIAÇÃO DOS MODELOS")
    print("="*70)
    
    # Divisão estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    
    print(f"\n📊 Divisão dos dados:")
    print(f"   Treino: {X_train.shape[0]} amostras")
    print(f"   Teste: {X_test.shape[0]} amostras")
    
    # Pesos de classe (para lidar com desbalanceamento)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    weights = dict(zip(np.unique(y_train), class_weights))
    print(f"   Pesos de classe: {weights}")
    
    # Definição dos modelos
    modelos = {
        "Regressão Logística": LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        ),
        "Árvore de Decisão": DecisionTreeClassifier(
            class_weight='balanced',
            max_depth=10,
            min_samples_split=20,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight='balanced',
            max_depth=15,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )
    }
    
    resultados = {}
    matrizes = {}
    predictions = {}
    probabilities = {}
    
    for nome, modelo in modelos.items():
        print(f"\n{'─'*70}")
        print(f"🔄 Treinando: {nome}...")
        modelo.fit(X_train, y_train)
        pred = modelo.predict(X_test)
        pred_proba = modelo.predict_proba(X_test)[:, 1]
        
        resultados[nome] = {
            "Acurácia": accuracy_score(y_test, pred),
            "Precisão": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1 Score": f1_score(y_test, pred, zero_division=0),
            "ROC AUC": roc_auc_score(y_test, pred_proba),
            "Relatório": classification_report(y_test, pred, zero_division=0)
        }
        
        matrizes[nome] = confusion_matrix(y_test, pred)
        predictions[nome] = pred
        probabilities[nome] = pred_proba
        
        print(f"   Acurácia: {resultados[nome]['Acurácia']:.4f}")
        print(f"   Precisão: {resultados[nome]['Precisão']:.4f}")
        print(f"   Recall:   {resultados[nome]['Recall']:.4f}")
        print(f"   F1 Score: {resultados[nome]['F1 Score']:.4f}")
        print(f"   ROC AUC:  {resultados[nome]['ROC AUC']:.4f}")
    
    return modelos, resultados, matrizes, predictions, probabilities, X_test, y_test


# =============================================================================
# FUNÇÃO: COMPARAÇÃO DE MODELOS
# =============================================================================
def plot_model_comparison(resultados):
    """
    Gera gráfico de comparação entre modelos.
    """
    print("\n" + "="*70)
    print("📊 COMPARAÇÃO DE MODELOS")
    print("="*70)
    
    # Preparar dados
    metrics = ['Acurácia', 'Precisão', 'Recall', 'F1 Score', 'ROC AUC']
    models = list(resultados.keys())
    
    data = {metric: [resultados[model][metric] for model in models] 
            for metric in metrics}
    df_comparison = pd.DataFrame(data, index=models)
    
    # Gráfico de barras agrupadas
    fig, ax = plt.subplots(figsize=(12, 6))
    df_comparison.plot(kind='bar', ax=ax, width=0.8)
    ax.set_title('Comparação de Performance dos Modelos', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Modelo', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim([0, 1])
    ax.legend(title='Métrica', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/13_model_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Figura 13: Comparação de Modelos")


# =============================================================================
# FUNÇÃO: MATRIZES DE CONFUSÃO
# =============================================================================
def plot_confusion_matrices(matrizes):
    """
    Plota matrizes de confusão de todos os modelos.
    
    INSIGHT DO TCC: A Regressão Logística teve o menor número de falsos 
    negativos (21), sendo mais eficaz para identificar colaboradores em risco.
    """
    print("\n" + "="*70)
    print("📊 MATRIZES DE CONFUSÃO")
    print("="*70)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, (nome, matriz) in enumerate(matrizes.items()):
        sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues',
                    ax=axes[idx], cbar=False,
                    annot_kws={'size': 14, 'weight': 'bold'})
        axes[idx].set_title(f'Matriz de Confusão\n{nome}', 
                           fontsize=11, fontweight='bold')
        axes[idx].set_ylabel('Real', fontsize=10)
        axes[idx].set_xlabel('Predito', fontsize=10)
        axes[idx].set_xticklabels(['Ativo', 'Desligado'])
        axes[idx].set_yticklabels(['Ativo', 'Desligado'])
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/14_confusion_matrices.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Figura 14: Matrizes de Confusão")
    
    # Imprimir análise de falsos negativos
    print("\n   📊 Análise de Falsos Negativos (FN):")
    for nome, matriz in matrizes.items():
        fn = matriz[1, 0]  # Falsos negativos
        print(f"      {nome}: {fn} FN")


# =============================================================================
# FUNÇÃO: CURVAS ROC
# =============================================================================
def plot_roc_curves(y_test, probabilities):
    """
    Plota curvas ROC de todos os modelos.
    """
    print("\n" + "="*70)
    print("📈 CURVAS ROC")
    print("="*70)
    
    plt.figure(figsize=(10, 8))
    
    colors = [COLORS['attrition'], COLORS['neutral'], COLORS['active']]
    
    for idx, (nome, y_proba) in enumerate(probabilities.items()):
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=colors[idx], lw=2,
                label=f'{nome} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', 
             label='Chance (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos', fontsize=12, fontweight='bold')
    plt.ylabel('Taxa de Verdadeiros Positivos', fontsize=12, fontweight='bold')
    plt.title('Curvas ROC - Comparação de Modelos', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/15_roc_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Figura 15: Curvas ROC")


# =============================================================================
# FUNÇÃO: COEFICIENTES DA REGRESSÃO LOGÍSTICA
# =============================================================================
def plot_logistic_regression_coefficients(modelos, feature_names):
    """
    Plota os coeficientes da Regressão Logística.
    
    PRINCIPAIS ACHADOS DO TCC:
    - FATORES DE RISCO (+): YearsSinceLastPromotion, OverTime, 
      Sales Representative, viagens frequentes
    - FATORES PROTETIVOS (-): TotalWorkingYears, YearsAtCompany,
      JobSatisfaction, YearsWithCurrManager
    """
    print("\n" + "="*70)
    print("📊 COEFICIENTES DA REGRESSÃO LOGÍSTICA")
    print("="*70)
    
    # Obter coeficientes
    lr_coef = pd.Series(
        modelos["Regressão Logística"].coef_[0],
        index=feature_names
    )
    
    # Top 15 positivos e Top 15 negativos
    top_positive = lr_coef.nlargest(15)
    top_negative = lr_coef.nsmallest(15)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Coeficientes positivos (aumentam risco)
    top_positive.sort_values().plot(kind='barh', color=COLORS['attrition'], ax=ax1)
    ax1.set_title('Top 15 Features que AUMENTAM Risco de Attrition',
                  fontsize=11, fontweight='bold')
    ax1.set_xlabel('Coeficiente (log-odds)')
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    
    # Coeficientes negativos (reduzem risco)
    top_negative.sort_values().plot(kind='barh', color=COLORS['active'], ax=ax2)
    ax2.set_title('Top 15 Features que REDUZEM Risco de Attrition',
                  fontsize=11, fontweight='bold')
    ax2.set_xlabel('Coeficiente (log-odds)')
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/16_logistic_coefficients.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Figura 16: Coeficientes da Regressão Logística")
    
    # Imprimir top 5 de cada
    print("\n   🔴 TOP 5 FATORES DE RISCO:")
    for feat, coef in top_positive.head().items():
        print(f"      • {feat}: {coef:.4f}")
    
    print("\n   🟢 TOP 5 FATORES PROTETIVOS:")
    for feat, coef in top_negative.head().items():
        print(f"      • {feat}: {coef:.4f}")


# =============================================================================
# FUNÇÃO: FEATURE IMPORTANCE
# =============================================================================
def plot_feature_importance(modelos, feature_names):
    """
    Plota importância de features para Random Forest e Árvore de Decisão.
    """
    print("\n" + "="*70)
    print("🎯 IMPORTÂNCIA DAS FEATURES")
    print("="*70)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Árvore de Decisão
    tree_importance = pd.Series(
        modelos["Árvore de Decisão"].feature_importances_,
        index=feature_names
    ).nlargest(15).sort_values()
    
    tree_importance.plot(kind='barh', color=COLORS['neutral'], ax=ax1)
    ax1.set_title('Top 15 Features - Árvore de Decisão', 
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('Importância')
    
    # Random Forest
    rf_importance = pd.Series(
        modelos["Random Forest"].feature_importances_,
        index=feature_names
    ).nlargest(15).sort_values()
    
    rf_importance.plot(kind='barh', color=COLORS['active'], ax=ax2)
    ax2.set_title('Top 15 Features - Random Forest', 
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('Importância')
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/17_feature_importance.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Figura 17: Importância das Features")


# =============================================================================
# FUNÇÃO: SALVAR RELATÓRIOS
# =============================================================================
def save_reports(resultados, best_model):
    """
    Salva relatórios de métricas e resumo executivo.
    """
    print("\n" + "="*70)
    print("💾 SALVANDO RELATÓRIOS")
    print("="*70)

    # Salvar métricas em arquivo
    metrics_path = os.path.join(REPORTS_DIR, "metricas_modelos.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("="*70 + "\n")
        f.write("MÉTRICAS DE AVALIAÇÃO DOS MODELOS\n")
        f.write("TCC - Predição de Rotatividade Voluntária\n")
        f.write("="*70 + "\n\n")

        for nome, metricas in resultados.items():
            f.write(f"\n{'='*70}\n")
            f.write(f"Modelo: {nome}\n")
            f.write(f"{'='*70}\n")
            f.write(f"Acurácia:  {metricas['Acurácia']:.4f}\n")
            f.write(f"Precisão:  {metricas['Precisão']:.4f}\n")
            f.write(f"Recall:    {metricas['Recall']:.4f}\n")
            f.write(f"F1 Score:  {metricas['F1 Score']:.4f}\n")
            f.write(f"ROC AUC:   {metricas['ROC AUC']:.4f}\n\n")
            f.write("Relatório de Classificação:\n")
            f.write(f"{metricas['Relatório']}\n")

        f.write(f"\n{'='*70}\n")
        f.write(f"🏆 MODELO FINAL ESCOLHIDO: {best_model}\n")
        f.write(f"{'='*70}\n")
        f.write(f"\nCritério de Seleção:\n")
        f.write(f"- Maior Recall: {resultados[best_model]['Recall']:.4f}\n")
        f.write(f"- Menor número de falsos negativos\n")
        f.write(f"- Melhor equilíbrio entre métricas\n")
        f.write(f"\nJustificativa:\n")
        f.write(f"Em contextos de retenção de talentos, é mais crítico identificar\n")
        f.write(f"colaboradores que efetivamente irão se desligar (minimizar FN)\n")
        f.write(f"do que maximizar a acurácia geral.\n")

    print(f"   ✅ Métricas: {metrics_path}")

    # Criar resumo executivo
    summary_path = os.path.join(REPORTS_DIR, "resumo_executivo.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("="*70 + "\n")
        f.write("RESUMO EXECUTIVO - PREDIÇÃO DE ATTRITION\n")
        f.write("="*70 + "\n\n")

        f.write("🎯 OBJETIVO\n")
        f.write("-" * 70 + "\n")
        f.write("Desenvolver modelo preditivo para identificar colaboradores em\n")
        f.write("risco de desligamento voluntário, subsidiando estratégias de\n")
        f.write("retenção de talentos em empresas de tecnologia.\n\n")

        f.write("📊 MODELO SELECIONADO\n")
        f.write("-" * 70 + "\n")
        f.write(f"Algoritmo: {best_model}\n")
        f.write(f"Recall: {resultados[best_model]['Recall']:.2%}\n")
        f.write(f"Acurácia: {resultados[best_model]['Acurácia']:.2%}\n")
        f.write(f"F1 Score: {resultados[best_model]['F1 Score']:.4f}\n")
        f.write(f"ROC AUC: {resultados[best_model]['ROC AUC']:.4f}\n\n")

        f.write("🔴 PRINCIPAIS FATORES DE RISCO\n")
        f.write("-" * 70 + "\n")
        f.write("1. Tempo desde última promoção (estagnação na carreira)\n")
        f.write("2. Realização frequente de horas extras (sobrecarga)\n")
        f.write("3. Cargo: Sales Representative\n")
        f.write("4. Viagens frequentes a trabalho\n")
        f.write("5. Alto número de empresas anteriores\n\n")

        f.write("🟢 PRINCIPAIS FATORES PROTETIVOS\n")
        f.write("-" * 70 + "\n")
        f.write("1. Experiência profissional total (TotalWorkingYears)\n")
        f.write("2. Tempo na empresa atual (YearsAtCompany)\n")
        f.write("3. Satisfação no trabalho (JobSatisfaction)\n")
        f.write("4. Tempo com gestor atual (YearsWithCurrManager)\n")
        f.write("5. Envolvimento no trabalho (JobInvolvement)\n\n")

        f.write("💡 RECOMENDAÇÕES ESTRATÉGICAS\n")
        f.write("-" * 70 + "\n")
        f.write("1. Implementar políticas claras de progressão na carreira\n")
        f.write("2. Monitorar e limitar horas extras excessivas\n")
        f.write("3. Fortalecer práticas de liderança próxima e empática\n")
        f.write("4. Criar programas de reconhecimento e desenvolvimento\n")
        f.write("5. Realizar acompanhamento periódico de satisfação\n\n")

        f.write("📈 IMPACTO ESPERADO\n")
        f.write("-" * 70 + "\n")
        f.write("• Identificação precoce de 70% dos colaboradores em risco\n")
        f.write("• Redução de custos com recrutamento e treinamento\n")
        f.write("• Melhoria no clima organizacional\n")
        f.write("• Retenção de talentos críticos\n")

    print(f"   ✅ Resumo executivo: {summary_path}")


# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================
def main():
    """
    Executa o pipeline completo de análise preditiva.
    
    Pipeline baseado no processo KDD:
    1. Seleção e compreensão dos dados
    2. Pré-processamento e transformação
    3. Mineração de dados (modelagem)
    4. Interpretação dos resultados
    """
    print("\n" + "="*70)
    print("🚀 PIPELINE COMPLETO - EMPLOYEE ATTRITION PREDICTION")
    print("="*70)
    print("   Projeto: TCC - Predição de Rotatividade Voluntária")
    print("   Instituição: USP/ECA - MBA Business Intelligence & Analytics")
    print("   Autora: Bettina Pietrobon Taucer Araujo")
    print("   Dataset: IBM HR Analytics")
    print("   Data: " + datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    print("="*70)
    
    try:
        # 1. Download do dataset
        dataset_path = download_dataset()
        
        # 2. Carregar dados
        df = load_data(dataset_path)
        
        # 3. EDA Completa
        exploratory_analysis_complete(df)
        
        # 4. Matriz de Correlação
        plot_correlation_matrix(df)
        
        # 5. Pré-processamento
        X_scaled, y, preprocessor, feature_names = preprocess(df)
        
        # 6. PCA Completo
        pca, scores = run_pca_complete(X_scaled, y)
        
        # 7. Treinamento de modelos
        modelos, resultados, matrizes, predictions, probabilities, X_test, y_test = \
            train_models(X_scaled, y)
        
        # 8. Comparação de modelos
        plot_model_comparison(resultados)
        
        # 9. Matrizes de confusão
        plot_confusion_matrices(matrizes)
        
        # 10. Curvas ROC
        plot_roc_curves(y_test, probabilities)
        
        # 11. Coeficientes da Regressão Logística
        plot_logistic_regression_coefficients(modelos, feature_names)
        
        # 12. Feature Importance
        plot_feature_importance(modelos, feature_names)
        
        # 13. Selecionar melhor modelo (baseado em Recall)
        best_model = max(resultados.items(), key=lambda x: x[1]["Recall"])[0]
        
        print("\n" + "="*70)
        print(f"🏆 MODELO FINAL SELECIONADO: {best_model}")
        print("="*70)
        print(f"   Recall:    {resultados[best_model]['Recall']:.4f} ⭐")
        print(f"   Acurácia:  {resultados[best_model]['Acurácia']:.4f}")
        print(f"   Precisão:  {resultados[best_model]['Precisão']:.4f}")
        print(f"   F1 Score:  {resultados[best_model]['F1 Score']:.4f}")
        print(f"   ROC AUC:   {resultados[best_model]['ROC AUC']:.4f}")
        print("\n   Critério: Maior Recall + Menor número de falsos negativos")
        print("="*70)
        
        # 14. Salvar relatórios
        save_reports(resultados, best_model)
        
        print("\n" + "="*70)
        print("✅ PIPELINE CONCLUÍDO COM SUCESSO!")
        print("="*70)
        print(f"\n📊 Total de visualizações geradas: 17 figuras")
        print(f"\n📂 Resultados disponíveis em:")
        print(f"   📊 Figuras: /{FIGURES_DIR}/")
        print(f"   📄 Relatórios: /{REPORTS_DIR}/")
        print(f"   💾 Dados: /{DATA_DIR}/")
        print("\n" + "="*70)
        print("📚 Para mais informações, consulte o TCC completo.")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERRO: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
