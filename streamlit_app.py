# =========================================
# Dashboard pour l'octroi de crédits bancaires
# Author: Fatima Meguellati
# Last Modified: 16 Aout 2022
# =========================================
# Command to execute script locally: streamlit run app.py

import joblib
import streamlit as st
from lime import lime_tabular
import streamlit.components.v1 as components
import pickle
import matplotlib.pyplot as plt
import requests
import seaborn as sns
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go

#On récupère notre fichier clients pour obtenir les informations descriptives du client à prévoir
file_clients_descr = open("clients_test_descr.pkl", "rb") #fichier client initiale
donnees_clients_descr = pickle.load(file_clients_descr)
file_clients_descr.close()

#On récupère notre fichier clients pour obtenir les informations descriptives des clients de la base test pour comparaison
file_clients_descr2 = open("clients_train_descr.pkl", "rb") #fichier client initiale
donnees_clients_train_descr = pickle.load(file_clients_descr2)
file_clients_descr2.close()

#url = 'https://ocp7apicredit.herokuapp.com'

# Set FastAPI endpoints : un pour les prédictions, un autre pour les explications
#endpoint_predict = 'http://127.0.0.1:8000/predict'
endpoint_predict = 'https://ocp7apicredit.herokuapp.com/predict' # Specify this path for Heroku deployment

#endpoint_lime = 'http://127.0.0.1:8000/lime'
endpoint_lime = 'https://ocp7apicredit.herokuapp.com/lime' # Specify this path for Heroku deployment

#endpoint_client = 'http://127.0.0.1:8000/client'
endpoint_client = 'https://ocp7apicredit.herokuapp.com/client' # Specify this path for Heroku deployment

#endpoint_client_data = 'http://127.0.0.1:8000/clientdata'
endpoint_client_data = 'https://ocp7apicredit.herokuapp.com/clientdata' # Specify this path for Heroku deployment

#endpoint_client_graph = 'http://127.0.0.1:8000/graphs'
endpoint_client_graph = 'https://ocp7apicredit.herokuapp.com/graphs' # Specify this path for Heroku deployment

# Mise en page de l'application streamlit
st.set_page_config(
    page_title="Dashboard",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        #'Get Help': 'https://www.extremelycoolapp.com/help',
        #'Report a bug': "https://www.extremelycoolapp.com/bug",
        #'About': "# This is a header. This is an *extremely* cool app!"
    }
)

# On définit ci-dessous nos variables de session pour maitriser le contrôle des différents boutons / liste déroulante...
if 'numclient' not in st.session_state: #numéro de client par défaut = 0
    st.session_state['numclient'] = 0

if 'rerunlime' not in st.session_state: # évite de relancer le lime à chaque action sur la page
    st.session_state['rerunlime'] = False

if 'output_lime' not in st.session_state: # sauvegarde les données du lime si déjà exécutées
    st.session_state['output_lime'] = ''

liste_client = pd.concat([donnees_clients_descr['SK_ID_CURR'],pd.Series([0])]) # pour la liste déroulante, on créée un client fictif '0' où rien ne se passe

def rerun():
    st.session_state['rerunlime'] = True # A chaque fois que l'on change de client, on relance le lime


with st.sidebar:
    
    NUM_CLIENT = st.selectbox("Numéro du client", options=liste_client, key = 'numclient',
                                help="Entrez le numéro de client de la base de données clients", on_change=rerun)

    client_json = {'num_client': NUM_CLIENT}

    client_valide = requests.post(endpoint_client, json=client_json,
                                timeout=8000)

st.header('Dashboard pour l\'octroi de crédits bancaires')

tab1, tab2, tab3= st.tabs(["Prévision", "Analyses comparatives", "Onglet test"])

with tab1:
    #st.header('Dashboard pour l\'octroi de crédits bancaires')

    #Lorsqu'on clique sur le bouton on obtient les prédictions
    if client_valide.json()[0]:
        with st.spinner('Prediction in Progress. Please Wait...'):
            previsions = requests.post(endpoint_predict, json=client_json, timeout=8000)

        #1er bloc qui contient les résultats de la prévision et données descriptives du client
        container_prev = st.empty()
        with container_prev.container():
            #On créé deux colonnes, une avec le résultat prévision et l'autre avec données descriptives
            col1, col2 = st.columns([0.75, 1], gap="medium")

            #1ere colonne avec résultats prévisions
            with col1:
                if previsions.json()["resultat"] == 0:
                    indicateur_pret = 'green'
                    message="Bravo, votre demande de crédit peut être acceptée"
                else:
                    indicateur_pret = 'red'
                    message="Malheureusement, votre demande de crédit ne peut être acceptée"
                st.write(message)                
                #Affichage jauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = previsions.json()["score"]*100,
                    delta = {'reference': 50, 'increasing':{'color': "red"}, 'decreasing':{'color': "green"}},
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    number = {'suffix': '%'},
                    gauge = {'axis': {'range': [None, 100]},
                            'steps' : [
                            {'range': [0, 40], 'color': "green"},
                            {'range': [40, 50], 'color': "orange"},
                            {'range': [50, 100], 'color': "red"}],
                            'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 51},
                            'bar': {'color': "gray"}},
                    title = {'text': "Score de faillite"}))

                #fig.show()
                st.plotly_chart(fig, use_container_width=True)
            
            #2ème colonne avec données descriptives
            with col2:
                choix_attributs = st.multiselect("Sélectionnez les attributs du client à afficher", donnees_clients_descr.columns )
                donnees_clients = requests.post(endpoint_client_data, json=client_json, timeout=8000)
                info_client = pd.read_json(donnees_clients.json()[0], orient='records')
                
                # on n'affiche que les attributs sélectionnés dans le sidebar
                info_client_choixattributs = info_client.loc[:, choix_attributs]
                st.dataframe(info_client_choixattributs)
                


        #2ème bloc qui contient les explications de la prévision avec un expander pour faire patienter l'utilisateur le temps du chargement
        container_explain = st.empty()
        with container_explain.expander("Cliquez ici pour obtenir des explications concernant cette décision"):
            if st.session_state.rerunlime == True:
                with st.spinner('Veuillez patienter, nous récupérons des données supplémentaires pour expliquer la décision...'):
                    st.session_state['output_lime'] = requests.post(endpoint_lime, json=client_json, timeout=8000)
                    st.session_state['rerunlime'] = False
            
            components.html(st.session_state['output_lime'].json()[0], height=200)

    else:
        st.warning(
            "Veuillez sélectionner un numéro de client dans le panneau latéral gauche pour obtenir des informations concernant la demande d'octroi de prêt")
        #obtain_pred = st.button('Cliquer ici pour connaitre la décision d\'accorder le prêt ou non', disabled=True)



with tab3:
    if client_valide.json()[0]:
        st.write("Onglet pour réaliser quelques tests supplémentaires")

    else:
        st.warning(
            "Veuillez sélectionner un numéro de client dans le panneau latéral gauche pour obtenir des informations concernant la demande d'octroi de prêt")

with tab2:
    if client_valide.json()[0]:
    
        #On récupère toutes les features possibles, et l'utilisateur peut en choisir un certain nombre pour obtenir des boxplots
        features_choisies = st.multiselect("Choisissez les variables", donnees_clients_descr.columns)

        #On prépare un dataframe qui regroupe toutes les données : le client à comparer ainsi que les données test avec qui on compare
        # 1. On créé une colonne avec la target pour notre client à comparer ainsi qu'une colonne target categorielle
        monclient = donnees_clients_descr.loc[donnees_clients_descr['SK_ID_CURR'] == NUM_CLIENT]
        monclient['TARGET'] = [previsions.json()["resultat"]]
        
        # 2. On concatène nos deux databases pour afficher toutes les données dans le même boxplot
        mesdonneesclients = pd.concat([donnees_clients_train_descr,monclient])

        # 3. On créé une colonne avec la target categorielle
        mesdonneesclients['TARGET_cat'] = mesdonneesclients['TARGET'].astype('category')

        
        # On définit les éléments de mise en page des boxplots
        # 1. On définit le nombre de boxplots par ligne (max 3 à 4 sinon cela devient illisible)
        meanprops = {'marker':'o', 'markeredgecolor':'black',
                'markerfacecolor':'green'}
        
        # 2. nombre de boxplots par ligne, max 3-4 sinon illisible
        longueur_ligne = 3 

        # 3. On calcule le bon nombre de lignes de la grille
        if len(features_choisies)%longueur_ligne == 0: #Si le nombre de variables est un multiple de longueur_ligne
            nb_lignes = len(features_choisies)//longueur_ligne # nblignes de la grille = quotient div euclidienne nb de variables par longueur ligne
        else:
            nb_lignes = len(features_choisies)//longueur_ligne+1 # s'il y a un reste non nul, alors on rajoute une ligne
        
        #st.write(mesdonneesclients.loc[mesdonneesclients['SK_ID_CURR']==NUM_CLIENT])
        
        info_comparatives = st.button("Obtenir des infos")

        if info_comparatives:
            fig, ax = plt.subplots()
            
            for i in range(len(features_choisies)):
                plt.subplot(nb_lignes,longueur_ligne,i+1)
                ax = sns.boxplot(y="TARGET_cat", x=features_choisies[i], showmeans=True, 
                                meanprops=meanprops, data=mesdonneesclients, showfliers = False)
                ax = sns.swarmplot(y="TARGET_cat", x=features_choisies[i], data=mesdonneesclients.loc[mesdonneesclients['SK_ID_CURR']==NUM_CLIENT], 
                                    color ='firebrick',size=8, linewidth=2, edgecolor='black')

            fig.tight_layout()

            st.pyplot(fig)

    else:
        st.warning(
            "Veuillez sélectionner un numéro de client dans le panneau latéral gauche pour obtenir des informations concernant la demande d'octroi de prêt")
