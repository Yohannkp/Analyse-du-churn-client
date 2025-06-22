import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="Dashboard Churn Client", layout="wide")
st.title("Dashboard interactif d'analyse du churn client")

# Chargement des données
data = pd.read_excel("data/Dataset_Clients_Churn.xlsx")
st.sidebar.header("Filtres")

# Filtres interactifs
gender = st.sidebar.multiselect("Genre", options=data['Gender'].unique(), default=list(data['Gender'].unique()))
contract = st.sidebar.multiselect("Type de contrat", options=data['ContractType'].unique(), default=list(data['ContractType'].unique()))

filtered_data = data[(data['Gender'].isin(gender)) & (data['ContractType'].isin(contract))]

# KPIs visuels
col1, col2, col3 = st.columns(3)
col1.metric("Nombre de clients", len(filtered_data))
col2.metric("Taux de churn", f"{filtered_data['Churn'].mean()*100:.1f}%")
col3.metric("Dépense mensuelle moyenne", f"{filtered_data['MonthlySpend'].mean():.0f} €")

st.markdown("---")

tabs = st.tabs(["Exploration générale", "Exploration avancée"])

with tabs[0]:
    st.subheader("Aperçu des données filtrées")
    st.dataframe(filtered_data.head())

    # Statistiques descriptives
    st.subheader("Statistiques descriptives")
    st.write(filtered_data.describe())

    # Visualisations
    st.subheader("Visualisations univariées")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Distribution de l'âge")
        fig, ax = plt.subplots()
        sns.histplot(filtered_data['Age'], kde=True, ax=ax)
        st.pyplot(fig)
        st.info("**Interprétation :** La distribution de l'âge permet d'identifier les segments d'âge les plus représentés et leur lien potentiel avec le churn.")
    with col2:
        st.write("Distribution du churn")
        fig, ax = plt.subplots()
        sns.countplot(x='Churn', data=filtered_data, ax=ax)
        st.pyplot(fig)
        st.info("**Interprétation :** Visualisez le déséquilibre éventuel entre churn et non churn dans l'échantillon filtré.")

    st.subheader("Analyse bivariée")
    fig, ax = plt.subplots()
    sns.boxplot(x='Churn', y='MonthlySpend', data=filtered_data, ax=ax)
    st.pyplot(fig)
    st.info("**Interprétation :** Les clients churn ont-ils tendance à dépenser plus ou moins que les autres ?")

    # Matrice de corrélation
    st.subheader("Matrice de corrélation")
    corr = filtered_data.select_dtypes(include=[np.number]).corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    st.info("**Interprétation :** Identifiez les variables les plus corrélées avec le churn et entre elles.")

    # Feature engineering (exemple)
    st.subheader("Feature engineering : création de variables")
    data_fe = filtered_data.copy()
    data_fe['TicketsPerTenure'] = data_fe['SupportTickets'] / data_fe['Tenure']
    data_fe['TicketsPerTenure'].replace([np.inf, -np.inf], np.nan, inplace=True)
    st.write(data_fe[['SupportTickets', 'Tenure', 'TicketsPerTenure']].head())

    # Modélisation simplifiée (Random Forest)
    st.subheader("Modélisation prédictive (Random Forest)")
    features = ['Age', 'Tenure', 'MonthlySpend', 'SupportTickets']
    X = data_fe[features].fillna(0)
    y = data_fe['Churn']

    if y.nunique() == 2 and len(X) > 10:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train_res, y_train_res)
        y_pred = clf.predict(X_test)
        st.write("Rapport de classification :")
        st.text(classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)
        st.info("**Interprétation :** Analysez la capacité du modèle à détecter les churns et les non churns.")
        # Bouton de téléchargement des clients à risque
        churn_clients = data_fe[data_fe['Churn'] == 1]
        csv = churn_clients.to_csv(index=False).encode('utf-8')
        st.download_button("Télécharger les clients à risque (churn)", csv, "clients_churn.csv", "text/csv")
    else:
        st.warning("Pas assez de données pour entraîner un modèle.")

    st.markdown("---")

    # Recommandations dynamiques
    st.markdown("**Recommandations opérationnelles :**")
    if filtered_data['Churn'].mean() > 0.2:
        st.warning("Taux de churn élevé : priorisez les actions sur les clients à forte dépense, faible ancienneté ou nombreux tickets de support.")
    else:
        st.success("Taux de churn maîtrisé : poursuivez les actions de fidélisation et surveillez les segments à risque.")
    st.markdown("- Cibler les clients avec beaucoup de tickets de support et une faible ancienneté.\n- Proposer des offres personnalisées aux clients à forte dépense.\n- Mettre en place un scoring de risque de churn pour prioriser les actions.")

    st.info("Ce dashboard reprend les principales analyses du notebook et permet une exploration interactive des données du churn client.")

with tabs[1]:
    st.header("Exploration avancée du churn")
    st.markdown("**Visualisations bivariées : variables numériques**")
    num_vars = ['Age', 'Tenure', 'MonthlySpend', 'SupportTickets']
    for var in num_vars:
        fig, ax = plt.subplots()
        sns.boxplot(x='Churn', y=var, data=filtered_data, ax=ax)
        st.pyplot(fig)
        st.info(f"Boxplot de {var} selon le churn.")
        fig, ax = plt.subplots()
        sns.histplot(data=filtered_data, x=var, hue='Churn', kde=True, element='step', stat='density', common_norm=False, ax=ax)
        st.pyplot(fig)
        st.info(f"Histogramme de {var} selon le churn.")
    st.markdown("**Visualisations bivariées : variables catégorielles**")
    cat_vars = ['ContractType', 'InternetService', 'Gender']
    for var in cat_vars:
        fig, ax = plt.subplots()
        sns.countplot(data=filtered_data, x=var, hue='Churn', ax=ax)
        st.pyplot(fig)
        st.info(f"Répartition du churn selon {var}.")
        st.write(f"Tableau croisé pour {var} :")
        st.dataframe(pd.crosstab(filtered_data[var], filtered_data['Churn'], margins=True))
